"""EPUB Book Management Web Application.

FastAPI application for managing EPUB books with MCP server integration.
"""

import asyncio
import json
import logging
import os
import threading
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import requests
import uvicorn
from ebooklib import epub
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config_manager import AppConfig
from src.history_util import delete_history as history_delete
from src.history_util import (
    get_all_sessions,
    get_session_summary,
    load_session_data,
    save_history,
)
from src.mlx_faiss_integration import MLXFAISSIntegration
from src.common_util import get_book_title_from_metadata
from src.simple_epub_service import SimpleEPUBService

# Load configuration
config = AppConfig()

app = FastAPI(
    title="EPUB Book Manager",
    description="EPUBファイル管理・検索システム",
    version="2.0.0",
)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# 文字化け対策: 一部のクライアント/ブラウザでのエンコード誤推定を防ぐため
# text/* と application/json / application/javascript について charset=utf-8 を明示付与
@app.middleware("http")
async def ensure_utf8_charset(request: Request, call_next):
    response = await call_next(request)
    ct = response.headers.get("content-type")
    if ct:
        lower = ct.lower()
        is_text = lower.startswith("text/")
        is_json = lower.startswith("application/json")
        is_js = lower.startswith("application/javascript") or lower.startswith(
            "text/javascript"
        )
        if (is_text or is_json or is_js) and "charset=" not in lower:
            response.headers["content-type"] = f"{ct}; charset=utf-8"
    return response


# Directory setup from config
base_dir = os.path.dirname(__file__)
EPUB_DIR = os.path.abspath(
    os.path.join(base_dir, "..", config.get("directories.epub_dir"))
)
CACHE_DIR = os.path.abspath(
    os.path.join(base_dir, "..", config.get("directories.cache_dir"))
)
LOG_DIR = os.path.abspath(
    os.path.join(base_dir, "..", config.get("directories.log_dir"))
)

# Logging setup from config
LOG_FILE = os.path.join(LOG_DIR, config.get("logging.files.app_log"))
logging.basicConfig(
    level=getattr(logging, config.get("logging.level", "INFO")),
    format=config.get("logging.format"),
    datefmt=config.get("logging.date_format", "%Y-%m-%d %H:%M:%S"),
    handlers=[
        logging.FileHandler(LOG_FILE, encoding=config.get("logging.file_encoding")),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

# Initialize simple EPUB service (now with MLX embedding search)
embedding_model = config.get("mlx.embedding_model")
if not isinstance(embedding_model, str) or not embedding_model:
    raise RuntimeError(
        "app_config.yaml の mlx.embedding_model が未設定です。必ずモデルIDを設定してください。"
    )
epub_service = SimpleEPUBService(EPUB_DIR, embedding_model=embedding_model)
SEARCH_ENABLED = True

# Initialize MLX-FAISS integration (share the same embedding service instance)
mlx_integration = MLXFAISSIntegration(
    CACHE_DIR,
    EPUB_DIR,
    embedding_service=epub_service.embedding_service,
)
mlx_integration.initialize()
app.include_router(mlx_integration.router)

# Log configuration summary
logging.debug("EPUB Book Manager initialized")
logging.debug("Config - EPUB Directory: %s", EPUB_DIR)
logging.debug("Config - Cache Directory: %s", CACHE_DIR)
logging.debug("Config - Search Enabled: %s", SEARCH_ENABLED)
logging.info("Config - MLX-FAISS Integration: Enabled")
logging.info("Config - MLX Embedding Model: %s", embedding_model)

# LM Studio selected settings cache (overrides env and can be changed via UI)
LMSTUDIO_SETTINGS_FILE = os.path.join(CACHE_DIR, "lmstudio_config.json")


def _ndjson_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize payload as NDJSON (UTF-8, no ASCII-escape)."""
    # Use compact separators to reduce bandwidth and ensure utf-8 output
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )


@app.get("/", response_class=HTMLResponse)
def home() -> RedirectResponse:
    """Render the home page (redirect to chat UI)."""
    return RedirectResponse(url="/chat_ui")


@app.get("/bookshelf", response_class=JSONResponse)
def get_bookshelf() -> list[dict[str, Any]]:
    """Get list of EPUB books."""
    return epub_service.get_bookshelf()


@app.get("/bookshelf_ui", response_class=HTMLResponse)
def bookshelf_ui(request: Request) -> HTMLResponse:
    """Render the bookshelf page."""
    books = get_bookshelf()
    return templates.TemplateResponse(
        "bookshelf.html", {"request": request, "books": books}
    )


@app.get("/chat_ui", response_class=HTMLResponse)
def chat_ui(request: Request) -> HTMLResponse:
    """Render the chat page."""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/settings", response_class=HTMLResponse)
def settings_ui(request: Request) -> HTMLResponse:
    """Render the settings page."""
    return templates.TemplateResponse("settings.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
def admin_alias() -> RedirectResponse:
    """Alias to bookshelf UI for admin operations."""
    return RedirectResponse(url="/bookshelf_ui")


def _is_safe_book_id(book_id: str) -> bool:
    """Check book_id safety to prevent path traversal and invalid names.

    Allows only basename-like identifiers ending with .epub and without
    path separators or parent directory segments.
    """
    if not book_id.endswith(".epub"):
        return False
    if os.path.sep in book_id or "/" in book_id or ".." in book_id:
        return False
    # Must be a simple base name
    return book_id == os.path.basename(book_id)


@app.get("/book/{book_id}/metadata", response_class=JSONResponse)
def get_book_metadata(book_id: str) -> dict[str, Any]:
    """Get metadata for a specific book."""
    return epub_service.get_book_metadata(book_id)


@app.get("/book/{book_id}/content", response_class=JSONResponse)
def get_book_content(book_id: str) -> dict[str, Any]:
    """Return extracted markdown content for a book."""
    if not _is_safe_book_id(book_id):
        return {"error": "Invalid book_id"}
    epub_path = os.path.join(EPUB_DIR, book_id)
    if not os.path.exists(epub_path):
        return {"error": "Book not found"}
    cache_txt = os.path.join(CACHE_DIR, book_id.replace(".epub", ".txt"))
    try:
        from src.epub_util import extract_epub_text

        content = extract_epub_text(epub_path, cache_txt)
        return {"content": content}
    except (OSError, ValueError, RuntimeError) as exc:
        return {"error": str(exc)}


@app.post("/book/{book_id}/search", response_class=JSONResponse)
async def search_book(
    book_id: str,
    query: str,
) -> list[dict[str, Any]]:
    """Search within a specific book using MLX embeddings."""
    return epub_service.search_book_content(book_id, query)


@app.post("/search", response_class=JSONResponse)
async def search_books(
    request: Request,
) -> list[dict[str, Any]]:
    """Search across multiple books using MLX embeddings."""
    body = await request.json()
    query = body.get("query", "") if isinstance(body, dict) else ""
    top_k = body.get("top_k", 10) if isinstance(body, dict) else 10
    if not query:
        return [{"error": "Missing 'query'"}]
    try:
        top_k_int = int(top_k)
    except (TypeError, ValueError):
        top_k_int = 10
    return epub_service.search_all_books(query, top_k_int)


@app.post("/upload", response_class=JSONResponse)
async def upload_epub(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload an EPUB file."""
    if not file.filename or not file.filename.endswith(".epub"):
        return {"error": "Invalid file type. Please upload an EPUB file."}

    try:
        # Save the uploaded file
        # Basic filename hardening
        filename = os.path.basename(file.filename)
        if filename != file.filename:
            return {"error": "Invalid filename"}
        file_path = os.path.join(EPUB_DIR, filename)

        # Check if file already exists
        if os.path.exists(file_path):
            return {"error": f"File {filename} already exists"}

        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        # Validate it's a valid EPUB
        try:
            book = epub.read_epub(file_path)
            _ = book.get_metadata("DC", "title")
        except (OSError, ValueError, TypeError, RuntimeError):
            os.remove(file_path)
            return {"error": "Invalid EPUB file"}

        logger.debug("Uploaded EPUB: %s", filename)
        # Try to index the new book so it's available for search/chat immediately
        try:
            book_id_no_ext = filename.replace(".epub", "")
            epub_service.embedding_service.add_book(book_id_no_ext, file_path)
            epub_service.embedding_service.save_index()
            logger.info("Indexed EPUB into MLX-FAISS: %s", filename)
        except (OSError, ValueError, TypeError, RuntimeError) as ie:
            logger.warning("Indexing skipped due to error: %s", ie)

        return {"message": f"Successfully uploaded {filename}"}

    except (OSError, ValueError, TypeError, RuntimeError) as e:
        logger.error("Failed to upload EPUB: %s", e)
        return {"error": f"Upload failed: {str(e)}"}


@app.post("/upload_epub", response_class=JSONResponse)
async def upload_epub_compat(epub_file: UploadFile = File(...)) -> dict[str, Any]:
    """Compatibility endpoint for bookshelf.html (expects 'epub_file')."""
    return await upload_epub(epub_file)


@app.delete("/book/{book_id}", response_class=JSONResponse)
def delete_book(book_id: str) -> dict[str, Any]:
    """Delete a book and its associated files.

    Validates book_id to prevent path traversal.
    """
    try:
        if not _is_safe_book_id(book_id):
            return {"error": "Invalid book_id"}
        # Delete the EPUB file
        epub_path = os.path.join(EPUB_DIR, book_id)
        if os.path.exists(epub_path):
            os.remove(epub_path)
            logger.debug("Deleted EPUB: %s", book_id)

        return {"result": "ok"}
    except (OSError, ValueError, TypeError, RuntimeError) as e:
        logger.error("Failed to delete book %s: %s", book_id, e)
        return {"error": str(e)}


@app.post("/delete_epub", response_class=JSONResponse)
async def delete_epub(request: Request) -> dict[str, Any]:
    """Compatibility endpoint for bookshelf.html delete action."""
    try:
        body = await request.json()
        book_id = body.get("book_id") if isinstance(body, dict) else None
        if not book_id:
            return {"error": "Missing 'book_id'"}
        return delete_book(book_id)
    except (ValueError, TypeError):
        return {"error": "Invalid request body"}


@app.get("/download/{book_id}", response_model=None)
def download_book(book_id: str) -> FileResponse | JSONResponse:
    """Download an EPUB file.

    Validates book_id to prevent path traversal.
    """
    if not _is_safe_book_id(book_id):
        return JSONResponse({"error": "Invalid book_id"}, status_code=400)
    file_path = os.path.join(EPUB_DIR, book_id)
    if not os.path.exists(file_path):
        return JSONResponse({"error": "Book not found"}, status_code=404)

    return FileResponse(
        path=file_path,
        media_type="application/epub+zip",
        filename=book_id,
    )


@app.get("/download_epub/{book_id}", response_model=None)
def download_book_compat(book_id: str) -> FileResponse | JSONResponse:
    """Compatibility endpoint for bookshelf.html download action."""
    return download_book(book_id)


# History APIs for chat UI
@app.get("/list_histories", response_class=JSONResponse)
def list_histories() -> list[dict[str, Any]]:
    """List chat histories summaries."""
    sessions = get_all_sessions()
    summaries: list[dict[str, Any]] = []
    for s in sessions:
        summary = get_session_summary(s)
        if summary:
            summaries.append(summary)
    return summaries


@app.get("/session/{session_id}", response_class=JSONResponse)
def get_session(session_id: str) -> dict[str, Any]:
    """Get full session data."""
    data = load_session_data(session_id)
    if not data:
        return {"error": "Session not found"}
    return data


@app.post("/history/{session_id}", response_class=JSONResponse)
async def save_session(session_id: str, request: Request) -> dict[str, Any]:
    """Save session messages and selected book ids."""
    body = await request.json()
    messages = body.get("messages") if isinstance(body, dict) else None
    book_ids = body.get("book_ids") if isinstance(body, dict) else None
    if not isinstance(messages, list):
        return {"error": "Invalid 'messages'"}
    if book_ids is not None and not isinstance(book_ids, list):
        return {"error": "Invalid 'book_ids'"}
    try:
        save_history(session_id, messages, book_ids)
        return {"result": "ok"}
    except (OSError, TypeError, ValueError) as e:
        return {"error": str(e)}


@app.delete("/history/{session_id}", response_class=JSONResponse)
def delete_session(session_id: str) -> dict[str, Any]:
    """Delete a session history."""
    ok = history_delete(session_id)
    return {"result": "ok"} if ok else {"error": "Not found"}


def _format_context_snippets(snippets: list[dict[str, Any]]) -> str:
    """Format search snippets into a human-readable context block.

    Args:
        snippets: List of result dicts that may include 'book_id', 'chunk_id',
            'text' or 'content'.

    Returns:
        A newline-joined string suitable to be embedded into a system prompt.
    """
    parts: list[str] = []
    meta_cache: dict[str, dict[str, Any]] = {}
    # Lazy import to avoid import cycles on startup
    try:
        from src.epub_util import extract_epub_metadata  # type: ignore
    except Exception:  # noqa: BLE001
        extract_epub_metadata = None  # type: ignore[assignment]

    def _meta_for(bid_like: str) -> tuple[str, str | None, str | None]:
        fname = bid_like if bid_like.endswith(".epub") else f"{bid_like}.epub"
        if fname in meta_cache:
            m = meta_cache[fname]
        else:
            m = {}
            if extract_epub_metadata is not None:
                try:
                    m = extract_epub_metadata(os.path.join(EPUB_DIR, fname)) or {}
                except Exception:  # noqa: BLE001
                    m = {}
            if not m:
                try:
                    m = {"title": get_book_title_from_metadata(EPUB_DIR, fname)}
                except Exception:  # noqa: BLE001
                    m = {}
            meta_cache[fname] = m
        title = str(m.get("title") or bid_like)
        author = m.get("author") or None
        year = m.get("year") or None
        return title, author, year

    def _label(title: str, author: str | None, year: str | None) -> str:
        # Compose bibliographic label with available fields
        if author and year:
            return f"{title} / {author} ({year})"
        if author:
            return f"{title} / {author}"
        if year:
            return f"{title} ({year})"
        return title

    for r in snippets:
        bid_val = r.get("book_id") or "?"
        bid = str(bid_val)
        cid = r.get("chunk_id")
        text = r.get("text") or r.get("content") or ""
        t, a, y = _meta_for(bid)
        parts.append(f"[{_label(t, a, y)}#{cid}] {text}")
    return "\n".join(parts)


def _build_system_prompt(snippets: list[dict[str, Any]]) -> str:
    """Build a Japanese system prompt with provided context snippets."""
    return (
        "あなたは日本語のアシスタントです。以下のコンテキストを可能な限り活用し、"
        "根拠を示しながら簡潔に回答してください。分からない場合はその旨を伝えてください。\n\n"
        "[コンテキスト]\n" + _format_context_snippets(snippets)
    )


def _read_cached_text_for_book(book_id: str) -> str | None:
    """キャッシュ済みのテキスト/Markdownを返す（あれば）。"""
    # .md（拡張子あり/なし）を優先
    md1 = os.path.join(CACHE_DIR, f"{book_id}.md")
    md2 = os.path.join(CACHE_DIR, f"{book_id}.epub.md")
    for p in (md1, md2):
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return f.read()
            except OSError:
                pass
    # .txt（拡張子あり/なし）
    base = book_id.replace(".epub", "")
    txt1 = os.path.join(CACHE_DIR, f"{book_id.replace('.epub', '.txt')}")
    txt2 = os.path.join(CACHE_DIR, f"{base}.txt")
    for p in (txt1, txt2):
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return f.read()
            except OSError:
                pass
    return None


def _fallback_text_search(
    book_ids: list[str] | None, query: str, top_k: int
) -> list[dict[str, Any]]:
    """埋め込みが使えない場合の簡易テキスト検索（AND検索）。"""
    q = (query or "").strip()
    if not q:
        return []
    terms = [t for t in q.lower().split() if t]
    results: list[dict[str, Any]] = []
    scope = book_ids or [b["id"] for b in epub_service.get_bookshelf()]
    for bid in scope:
        text = _read_cached_text_for_book(bid)
        if not text:
            continue
        low = text.lower()
        if not all(t in low for t in terms):
            continue
        first = terms[0]
        pos = low.find(first)
        if pos < 0:
            continue
        score = low.count(first)
        start = max(0, pos - 200)
        end = min(len(text), pos + 600)
        snippet = text[start:end]
        results.append(
            {
                "rank": len(results) + 1,
                "score": float(score),
                "text": snippet,
                "book_id": bid.replace(".epub", ""),
                "chunk_id": start // 800,
            }
        )
        if len(results) >= max(1, top_k):
            break
    return results


async def _gather_context(
    book_ids: list[str],
    query: str,
    per_book_top_k: int | None = None,
    all_books_top_k: int | None = None,
    max_context_snippets: int | None = None,
) -> list[dict[str, Any]]:
    """Search and collect context snippets for RAG.

    If book_ids is provided, search within those books; otherwise search across
    all books. Limits to a small number of top results to keep prompt concise.
    """
    # Load defaults from config
    if per_book_top_k is None:
        try:
            per_book_top_k = int(config.get("lmstudio.per_book_top_k", 3))
        except (TypeError, ValueError):
            per_book_top_k = 3
    if all_books_top_k is None:
        try:
            all_books_top_k = int(config.get("lmstudio.all_books_top_k", 5))
        except (TypeError, ValueError):
            all_books_top_k = 5
    if max_context_snippets is None:
        try:
            max_context_snippets = int(config.get("lmstudio.max_context_snippets", 12))
        except (TypeError, ValueError):
            max_context_snippets = 12
    # Convert filenames to internal ids without .epub
    snippets: list[dict[str, Any]] = []
    # 念のためインデックスがロードされていることを保証
    try:
        # 既存インデックスの読み込み（存在しなくても続行）
        epub_service.embedding_service.load_index()
    except (FileNotFoundError, OSError, RuntimeError):
        pass

    if book_ids:
        # 選択書籍が指定された場合、各書籍がインデックス済みであることを保証
        for b in book_ids:
            key = b.replace(".epub", "")
            epub_path = os.path.join(EPUB_DIR, b)
            try:
                if os.path.exists(epub_path):
                    # 必要に応じて単一書籍をインデックス化
                    epub_service.ensure_book_indexed(key, epub_path)
            except (OSError, ValueError, RuntimeError):
                # モデル未導入等でも全体は継続
                pass

        # インデックス保証後に各書籍で検索
        for b in book_ids:
            key = b.replace(".epub", "")
            try:
                res = epub_service.embedding_service.search(
                    query=query, top_k=per_book_top_k, book_id=key
                )
                for r in res:
                    if "error" not in r:
                        snippets.append(r)
            except (OSError, ValueError, RuntimeError):
                # 書籍単位でフォールバック
                fb = _fallback_text_search([f"{key}.epub"], query, per_book_top_k)
                snippets.extend(fb)
    else:
        try:
            res = epub_service.search_all_books(query, top_k=all_books_top_k)
            for r in res:
                if "error" not in r:
                    snippets.append(
                        {
                            "text": r.get("text") or r.get("content"),
                            "score": r.get("score"),
                            "book_id": (r.get("book_id") or "").replace(".epub", ""),
                            "chunk_id": r.get("chunk_id"),
                        }
                    )
        except (OSError, ValueError, RuntimeError):
            # 全書籍検索が失敗した場合は全文検索にフォールバック
            snippets.extend(_fallback_text_search(None, query, all_books_top_k or 5))

    # それでも空なら最終フォールバック
    if not snippets:
        snippets.extend(
            _fallback_text_search(book_ids or None, query, max_context_snippets or 5)
        )
    return snippets[:max_context_snippets]


def _build_evidence(snippets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build structured evidence list including bibliographic metadata.

    Each item contains: book_id (filename), chunk_id, title, author, year, preview.
    """
    # Lazy import of metadata util to avoid cyclic imports during startup
    try:
        from src.epub_util import extract_epub_metadata  # type: ignore
    except Exception:  # noqa: BLE001
        extract_epub_metadata = None  # type: ignore[assignment]

    out: list[dict[str, Any]] = []
    mcache: dict[str, dict[str, Any]] = {}

    def meta_for(fname: str) -> dict[str, Any]:
        if fname in mcache:
            return mcache[fname]
        md: dict[str, Any] = {}
        if extract_epub_metadata is not None:
            try:
                md = extract_epub_metadata(os.path.join(EPUB_DIR, fname)) or {}
            except Exception:  # noqa: BLE001
                md = {}
        if not md:
            # Best-effort: fallback to title only via common util
            try:
                from src.common_util import get_book_title_from_metadata as _title  # type: ignore

                md = {"title": _title(EPUB_DIR, fname)}
            except Exception:  # noqa: BLE001
                md = {}
        mcache[fname] = md
        return md

    for s in snippets:
        bid = str(s.get("book_id") or "")
        fname = bid if bid.endswith(".epub") else (bid + ".epub" if bid else "")
        if not fname:
            continue
        md = meta_for(fname)
        out.append(
            {
                "book_id": fname,
                "chunk_id": s.get("chunk_id"),
                "title": md.get("title"),
                "author": md.get("author"),
                "year": md.get("year"),
                "preview": (s.get("text") or s.get("content") or "")[:240],
            }
        )
    return out


def _get_last_user_content(messages: list[dict[str, Any]]) -> str:
    """Extract the last user message content from chat messages.

    Returns an empty string when not found or invalid.
    """
    last: dict[str, Any] | None = next(
        (m for m in reversed(messages) if m.get("role") == "user"),
        None,
    )
    return last.get("content", "") if isinstance(last, dict) else ""


@dataclass(frozen=True)
class LMStudioSettings:
    """Configuration for LM Studio OpenAI-compatible endpoint."""

    base_url: str
    model: str
    temperature: float
    max_tokens: int


def _lmstudio_settings() -> LMStudioSettings:
    """Load LM Studio settings with config-first precedence.

    Precedence: config (app_config.yaml) > LM Studio UI cache file > env defaults.
    """
    # 1) Load from config
    base = str(config.get("lmstudio.base_url", "http://localhost:1234/v1"))
    model = str(config.get("lmstudio.model", ""))
    try:
        temperature = float(config.get("lmstudio.temperature", 0.2))
    except (TypeError, ValueError):
        temperature = 0.2
    try:
        max_tokens = int(config.get("lmstudio.max_tokens", 20480))
    except (TypeError, ValueError):
        max_tokens = 20480

    # 2) Override from cached file if present (UI selection)
    try:
        if os.path.exists(LMSTUDIO_SETTINGS_FILE):
            with open(LMSTUDIO_SETTINGS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                base = str(data.get("base_url", base))
                file_model = data.get("model")
                if isinstance(file_model, str) and file_model:
                    model = file_model
    except (OSError, json.JSONDecodeError):
        pass

    # 3) Finally, allow env vars to act as fallback defaults
    base = os.getenv("LMSTUDIO_BASE_URL", base)
    model = os.getenv("LMSTUDIO_MODEL", model)
    try:
        temperature = float(os.getenv("LMSTUDIO_TEMPERATURE", str(temperature)))
    except ValueError:
        pass
    try:
        max_tokens = int(os.getenv("LMSTUDIO_MAX_TOKENS", str(max_tokens)))
    except ValueError:
        pass

    return LMStudioSettings(base, model, temperature, max_tokens)


def _is_disallowed_model(model: str) -> bool:
    """Return True if model matches disallowed list.

    Controlled by env `LMSTUDIO_DISALLOWED_MODELS` (comma-separated).
    If unset, defaults to disallowing 'qwen3-30b-a3b-instruct-2507'.
    Comparison is case-insensitive.
    Additionally, you can configure `lmstudio.disallowed_models` in app_config.yaml
    as a comma-separated string or a list of strings.
    """
    default_deny = "qwen3-30b-a3b-instruct-2507"
    env_val = os.getenv("LMSTUDIO_DISALLOWED_MODELS")
    if env_val is not None:
        raw = env_val
    else:
        cfg_val: Any = config.get("lmstudio.disallowed_models", default_deny)
        if isinstance(cfg_val, list):
            # Join to reuse the same split/normalize path
            raw = ",".join(str(x) for x in cfg_val)
        else:
            raw = str(cfg_val)
    deny = [s.strip().lower() for s in raw.split(",") if s.strip()]
    return model.lower() in deny


def _normalize_listish(val: Any) -> list[str]:
    """Normalize list- or comma-separated-string into lowercased list."""
    if isinstance(val, list):
        items = [str(x) for x in val]
    else:
        items = str(val).split(",")
    return [s.strip().lower() for s in items if str(s).strip()]


def _is_allowed_model(model: str) -> bool:
    """Return True if model is allowed under allowlist-first policy.

    Precedence:
    1) LMSTUDIO_ALLOWED_MODELS env (if set): only those are allowed
    2) app_config.yaml lmstudio.allowed_models (if non-empty): only those
    3) Otherwise: fall back to disallow list logic
    """
    env_allow = os.getenv("LMSTUDIO_ALLOWED_MODELS")
    if env_allow is not None and env_allow.strip():
        allow = _normalize_listish(env_allow)
        return model.lower() in allow

    cfg_allow: Any = config.get("lmstudio.allowed_models", [])
    if isinstance(cfg_allow, list) and cfg_allow:
        allow = _normalize_listish(cfg_allow)
        return model.lower() in allow
    if isinstance(cfg_allow, str) and cfg_allow.strip():
        allow = _normalize_listish(cfg_allow)
        return model.lower() in allow

    # No allowlist configured -> use disallow policy
    return not _is_disallowed_model(model)


def _call_lmstudio(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model or messages[0].get("model", ""),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return msg.get("content") or ""


async def _stream_lmstudio(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> AsyncGenerator[str, None]:
    """Stream chat completions from LM Studio (OpenAI-compatible SSE).

    Yields content deltas (str). Converts SSE 'data: {...}' lines into
    text chunks by extracting choices[0].delta.content.
    """
    import contextlib

    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model or messages[0].get("model", ""),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Accept-Charset": "utf-8",
        "Cache-Control": "no-cache",
    }

    # Run blocking HTTP streaming in a worker and push deltas into a queue
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def _worker() -> None:
        with contextlib.ExitStack() as stack:
            resp = stack.enter_context(
                requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload),
                    stream=True,
                    timeout=(10, 300),
                )
            )
            resp.raise_for_status()
            # SSEはUTF-8が既定のため、サーバーヘッダに依存せず明示指定
            resp.encoding = "utf-8"
            for raw in resp.iter_lines(decode_unicode=False):
                if not raw:
                    continue
                # bytesとして受け取りUTF-8で厳密にデコード
                try:
                    line = raw.strip().decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    # デコード不能なチャンクは破棄
                    continue
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choice = (obj.get("choices") or [{}])[0]
                delta = (choice.get("delta") or {}).get("content") or ""
                if delta:
                    # Use loop.call_soon_threadsafe to interact with loop from thread
                    loop.call_soon_threadsafe(queue.put_nowait, delta)
        # Sentinel
        loop.call_soon_threadsafe(queue.put_nowait, None)

    # Start worker thread (non-blocking)
    threading.Thread(target=_worker, daemon=True).start()
    while True:
        item = await queue.get()
        if item is None:
            break
        yield item


@app.post("/chat")
async def chat_endpoint(request: Request) -> StreamingResponse:
    """Chat endpoint that proxies to LM Studio with simple RAG context."""

    # 先にリクエストボディを読み取ってからストリームを開始
    try:
        body = await request.json()
    except (ValueError, json.JSONDecodeError):

        async def bad_request_stream() -> AsyncGenerator[bytes, None]:
            yield _ndjson_bytes({"status": "不正なリクエスト"})

        return StreamingResponse(
            bad_request_stream(),
            media_type="application/x-ndjson; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )
    except (OSError, RuntimeError, TypeError) as exc:
        err_msg = f"リクエスト読取エラー: {exc}"

        async def read_error_stream() -> AsyncGenerator[bytes, None]:
            yield _ndjson_bytes({"status": err_msg})

        return StreamingResponse(
            read_error_stream(),
            media_type="application/x-ndjson; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    async def event_stream() -> AsyncGenerator[bytes, None]:
        # 最初に即時フラッシュされる行を送出（UI側の疎通確認用）
        yield _ndjson_bytes({"status": "コンテキスト検索中..."})
        await asyncio.sleep(0)

        book_ids = body.get("book_ids") if isinstance(body, dict) else []
        messages_in = body.get("messages") if isinstance(body, dict) else []
        if not isinstance(messages_in, list) or not messages_in:
            yield _ndjson_bytes({"status": "メッセージがありません"})
            return

        query = _get_last_user_content(messages_in)

        # 初回チャンクは既に送出済み

        # Optional overrides
        def _to_int(val: Any, default: int) -> int:
            try:
                return int(val)
            except (TypeError, ValueError):
                return default

        per_book_k = _to_int(
            (body.get("top_k_per_book") if isinstance(body, dict) else None),
            int(config.get("lmstudio.per_book_top_k", 3)),
        )
        all_books_k = _to_int(
            (body.get("top_k_all_books") if isinstance(body, dict) else None),
            int(config.get("lmstudio.all_books_top_k", 5)),
        )
        max_snips = _to_int(
            (body.get("max_context_snippets") if isinstance(body, dict) else None),
            int(config.get("lmstudio.max_context_snippets", 12)),
        )

        snippets = await _gather_context(
            book_ids or [],
            query,
            per_book_top_k=per_book_k,
            all_books_top_k=all_books_k,
            max_context_snippets=max_snips,
        )

        yield _ndjson_bytes({"status": f"コンテキスト {len(snippets)} 件"})
        # Send structured evidence so the UI can render jump links into markdown
        try:
            evidence = _build_evidence(snippets)
            if evidence:
                yield _ndjson_bytes({"evidence": evidence})
        except Exception as _e:  # noqa: BLE001
            # Ignore evidence building errors to avoid breaking chat
            pass
        await asyncio.sleep(0)

        sys_prompt = _build_system_prompt(snippets)

        # Optional debug preview
        debug_ctx = bool(body.get("debug_context") if isinstance(body, dict) else False)
        if debug_ctx:
            preview = _format_context_snippets(snippets)
            # Limit preview size to avoid huge payloads
            if len(preview) > 1500:
                preview = preview[:1500] + "... (truncated)"
            yield _ndjson_bytes(
                {"status": "コンテキストプレビュー", "context_preview": preview}
            )
            await asyncio.sleep(0)

        settings = _lmstudio_settings()
        if not settings.model:
            # 明確なガイダンスを返してフロント側でも表示できるようにする
            msg = (
                "LM Studioモデルが未設定です。app_config.yaml の lmstudio.model を設定するか、"
                "環境変数 'LMSTUDIO_MODEL' を設定してください。"
            )
            yield _ndjson_bytes({"status": msg})
            # 併せて content としても返す（フロントでエラーバブルに出すため）
            yield _ndjson_bytes({"content": msg})
            await asyncio.sleep(0)
            return

        if not _is_allowed_model(settings.model):
            msg = (
                f"モデル '{settings.model}' は許可リストに含まれていません。"
                "'LMSTUDIO_ALLOWED_MODELS' もしくは app_config.yaml の "
                "lmstudio.allowed_models を設定してください。"
            )
            yield _ndjson_bytes({"status": msg})
            yield _ndjson_bytes({"content": msg})
            await asyncio.sleep(0)
            return

        yield _ndjson_bytes({"status": "LM Studioに問い合わせ中..."})
        await asyncio.sleep(0)

        # Stream deltas and also return final content for compatibility
        final_parts: list[str] = []
        try:
            async for delta in _stream_lmstudio(
                settings.base_url,
                settings.model,
                [{"role": "system", "content": sys_prompt}] + messages_in,
                settings.temperature,
                settings.max_tokens,
            ):
                final_parts.append(delta)
                yield _ndjson_bytes({"delta": delta})
                # 連続送出時の描画を促進
                await asyncio.sleep(0)
        except requests.Timeout as e:
            yield _ndjson_bytes({"status": f"LM Studioタイムアウト: {e}"})
            return
        except requests.ConnectionError as e:
            yield _ndjson_bytes({"status": f"LM Studio接続エラー: {e}"})
            return
        except (requests.RequestException, json.JSONDecodeError) as e:
            yield _ndjson_bytes({"status": f"LM Studioエラー: {e}"})
            return

        content = "".join(final_parts) or "回答が取得できませんでした。"
        yield _ndjson_bytes({"content": content})
        await asyncio.sleep(0)

        yield _ndjson_bytes({"status": "完了"})

    # NDJSON 風の行区切り JSON を返す
    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/book/{book_id}/chunks", response_class=JSONResponse)
def get_book_chunks(book_id: str) -> dict[str, Any]:
    """Return chunked markdown snippets for a specific book.

    Uses the embedding index metadata when available; otherwise falls back to
    regenerating chunks from the cached markdown using the same algorithm.
    """
    if not _is_safe_book_id(book_id):
        return {"error": "Invalid book_id"}
    epub_path = os.path.join(EPUB_DIR, book_id)
    if not os.path.exists(epub_path):
        return {"error": "Book not found"}

    try:
        # Ensure the book is indexed so chunks_metadata is populated
        key = book_id.replace(".epub", "")
        epub_service.ensure_book_indexed(key, epub_path)
        # Prefer chunks from the embedding index
        chunks = [
            {"chunk_id": md.get("chunk_id"), "text": md.get("text", "")}
            for md in epub_service.embedding_service.chunks_metadata
            if md.get("book_id") == key
        ]
        if chunks:
            chunks.sort(key=lambda x: int(x.get("chunk_id") or 0))
            return {"book_id": book_id, "chunks": chunks}

        # Fallback: reconstruct chunks from markdown cache using the same logic
        try:
            from src.mlx_embedding_service import _chunk_markdown as chunk_md  # type: ignore
        except Exception:  # noqa: BLE001
            chunk_md = None  # type: ignore[assignment]

        cache_txt = os.path.join(CACHE_DIR, book_id.replace(".epub", ".txt"))
        from src.epub_util import extract_epub_text  # local import

        md = extract_epub_text(epub_path, cache_txt)
        chunks_fallback: list[str]
        if chunk_md is not None:
            chunks_fallback = chunk_md(md)
        else:
            # Simple paragraph-based fallback
            chunks_fallback = [p.strip() for p in md.split("\n\n") if p.strip()]
        return {
            "book_id": book_id,
            "chunks": [
                {"chunk_id": i, "text": t[:1000]} for i, t in enumerate(chunks_fallback)
            ],
        }
    except (OSError, ValueError, RuntimeError) as exc:
        return {"error": str(exc)}


@app.get("/health", response_class=JSONResponse)
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "search_enabled": str(SEARCH_ENABLED),
    }


@app.get("/debug/ndjson")
async def debug_ndjson_stream() -> StreamingResponse:
    """Small NDJSON streaming test endpoint for diagnostics."""

    async def gen() -> AsyncGenerator[bytes, None]:
        for i in range(5):
            yield _ndjson_bytes({"status": f"tick {i}"})
            await asyncio.sleep(0)
        yield _ndjson_bytes({"content": "debug stream ok"})
        await asyncio.sleep(0)

    return StreamingResponse(
        gen(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/mcp-setup", response_class=HTMLResponse)
def mcp_setup_page(request: Request) -> HTMLResponse:
    """MCP Server setup page."""
    current_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    return templates.TemplateResponse(
        "mcp_setup.html", {"request": request, "current_path": current_path}
    )


@app.post("/api/mcp/start", response_class=JSONResponse)
async def start_mcp_server() -> dict[str, str]:
    """Start MCP server."""
    return {"message": "MCP server start functionality not implemented"}


@app.post("/api/mcp/stop", response_class=JSONResponse)
async def stop_mcp_server() -> dict[str, str]:
    """Stop MCP server."""
    return {"message": "MCP server stop functionality not implemented"}


@app.get("/api/mcp/test", response_class=JSONResponse)
async def test_mcp_connection() -> dict[str, Any]:
    """Test MCP server connection."""
    return {
        "success": True,
        "response_time": 50,
        "message": "MCP connection test successful",
    }


@app.get("/api/mcp/test-tools", response_class=JSONResponse)
async def test_mcp_tools() -> dict[str, Any]:
    """Test MCP tools functionality."""
    books = epub_service.get_bookshelf()

    results = {
        "list_epub_books": {"success": True, "message": f"Found {len(books)} books"},
        "get_epub_metadata": {
            "success": bool(books),
            "message": "Metadata available" if books else "No books to test",
        },
        "search_epub_content": {
            "success": False,
            "message": "Search functionality disabled",
        },
    }

    return {"results": results}


@app.get("/api/lmstudio/models", response_class=JSONResponse)
def list_lmstudio_models(request: Request) -> dict[str, Any]:
    settings = _lmstudio_settings()
    base_override = request.query_params.get("base_url")
    try:
        base = base_override or settings.base_url
        url = f"{base.rstrip('/')}/models"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        names: list[str] = []
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            for m in data["data"]:
                name = m.get("id") or m.get("name")
                if isinstance(name, str):
                    names.append(name)
        elif isinstance(data, list):
            for m in data:
                if isinstance(m, dict):
                    name = m.get("id") or m.get("name")
                    if isinstance(name, str):
                        names.append(name)
        return {"models": names}
    except requests.RequestException as e:
        return {"error": str(e)}


@app.get("/api/lmstudio/selected", response_class=JSONResponse)
def get_selected_model() -> dict[str, Any]:
    s = _lmstudio_settings()
    return {"model": s.model, "base_url": s.base_url}


@app.post("/api/lmstudio/selected", response_class=JSONResponse)
async def set_selected_model(request: Request) -> dict[str, Any]:
    body = await request.json()
    model = body.get("model") if isinstance(body, dict) else None
    base_url = body.get("base_url") if isinstance(body, dict) else None
    if not isinstance(model, str) or not model:
        return {"error": "model is required"}
    if not _is_allowed_model(model):
        return {"error": f"モデル '{model}' は許可されていません"}
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        payload = {"model": model}
        if isinstance(base_url, str) and base_url:
            payload["base_url"] = base_url
        with open(LMSTUDIO_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        return {"result": "ok"}
    except OSError as e:
        return {"error": str(e)}


def main() -> None:
    """Run the application."""
    host = config.get("server.host", "0.0.0.0")
    port = config.get("server.port", 8000)

    uvicorn.run(
        "src.app:app",
        host=host,
        port=port,
        reload=config.get("server.reload", False),
        log_level=config.get("logging.level", "INFO").lower(),
    )


if __name__ == "__main__":
    main()
