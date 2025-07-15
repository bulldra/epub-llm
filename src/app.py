"""EPUB LLM Web Application.

FastAPI application for EPUB-based RAG (Retrieval-Augmented Generation) system.
"""

import asyncio
import json
import logging
import multiprocessing
import os
from collections.abc import AsyncGenerator
from typing import Any

from ebooklib import epub
from fastapi import Body, FastAPI, File, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mlx_embeddings import load as load_embedding_model
from mlx_lm import load as load_mlx_model  # type: ignore[attr-defined]

from src.config_manager import AppConfig
from src.enhanced_epub_service import EnhancedChatService, EnhancedEPUBService
from src.epub_service import EPUBService
from src.llm_util import LLMManager
from src.rag_util import RAGManager
from src.smart_rag_util import SmartRAGManager

# Set multiprocessing start method for Python 3.12 compatibility
try:
    multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    pass  # Already set

# Load configuration
config = AppConfig()

app = FastAPI(
    title="EPUB LLM API", description="EPUBファイルを扱うためのAPI", version="1.0.0"
)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

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

# Model initialization from config
DEV_MODE = config.get("llm.dev_mode", False)

if DEV_MODE:
    MODEL, TOKENIZER = None, None
    logger.info("Running in development mode - models disabled")
else:
    MODEL_NAME = config.get("llm.model_name")
    MODEL, TOKENIZER = load_mlx_model(MODEL_NAME)
    logger.info("Loaded LLM model: %s", MODEL_NAME)

# Embedding model setup
EMBED_MODEL_NAME = config.get("llm.embedding_model_name")
if DEV_MODE:
    EMBED_MODEL, EMBED_TOKENIZER = None, None
    logger.info("Development mode - embedding model disabled")
else:
    EMBED_MODEL, EMBED_TOKENIZER = load_embedding_model(EMBED_MODEL_NAME)
    logger.info("Loaded embedding model: %s", EMBED_MODEL_NAME)

# Initialize managers
rag_manager = RAGManager(EMBED_MODEL, EMBED_TOKENIZER, CACHE_DIR, EPUB_DIR)
smart_rag_manager = SmartRAGManager(EMBED_MODEL, EMBED_TOKENIZER, CACHE_DIR, EPUB_DIR)
llm_manager = LLMManager(MODEL, TOKENIZER)

# Initialize services - both traditional and enhanced
epub_service = EPUBService(EPUB_DIR, CACHE_DIR, rag_manager)
enhanced_epub_service = EnhancedEPUBService(
    EPUB_DIR, CACHE_DIR, smart_rag_manager, llm_manager
)

# Use enhanced services for better performance
chat_service = EnhancedChatService(enhanced_epub_service, llm_manager)

# Log configuration summary
logging.info("EPUB-LLM Application initialized")
logging.info("Config - DEV_MODE: %s", DEV_MODE)
logging.info("Config - LLM Model: %s", MODEL_NAME if not DEV_MODE else "Disabled")
logging.info("Config - Embedding Model: %s", EMBED_MODEL_NAME)
logging.info("Config - EPUB Directory: %s", EPUB_DIR)
logging.info("Config - Cache Directory: %s", CACHE_DIR)


@app.get("/", response_class=HTMLResponse)
def chat_ui(request: Request) -> HTMLResponse:
    """Render the main chat UI."""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/list_histories", response_class=JSONResponse)
def list_histories() -> list[dict[str, Any]]:
    """Get all chat sessions with summaries."""
    return chat_service.get_all_sessions()


@app.get("/bookshelf", response_class=JSONResponse)
def bookshelf() -> list[dict[str, Any]]:
    """Get list of available EPUB books."""
    return enhanced_epub_service.get_bookshelf()


@app.get("/bookshelf_ui", response_class=HTMLResponse)
def bookshelf_ui(request: Request) -> HTMLResponse:
    """Render the bookshelf UI."""
    books = bookshelf()
    return templates.TemplateResponse(
        "bookshelf.html", {"request": request, "books": books}
    )


@app.post("/chat")
async def chat_stream(request: Request) -> StreamingResponse:
    """Stream chat responses using RAG and LLM."""
    try:
        logging.info("[CHAT] チャットリクエスト開始")
        body = await request.json()
        logging.info("[CHAT] リクエストボディ解析完了")
    except (ValueError, TypeError) as e:
        logging.error("[CHAT] リクエスト解析エラー: %s", e)
        raise

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            book_ids: list[str] = body.get("book_ids", [])
            messages: list[dict[str, str]] = body.get("messages", [])
            logging.info(
                "[CHAT] パラメータ取得完了 - book_ids: %d, messages: %d",
                len(book_ids),
                len(messages),
            )

            # Process chat request through service
            if book_ids:
                yield json.dumps({"status": "検索中...", "content": ""})
                await asyncio.sleep(0)

            chat_data = await chat_service.process_chat_request(messages, book_ids)

            # LLM生成
            status_msg = "生成開始..."
            yield json.dumps(
                {
                    "status": status_msg,
                    "content": "",
                }
            )
            await asyncio.sleep(0)

            # Format prompt
            prompt = chat_data["prompt"]
            logging.info("[LLM] プロンプト組み立て完了: %d文字", len(prompt))
            truncated_prompt = prompt[:300] + "..." if len(prompt) > 300 else prompt
            logging.debug("[LLM] Final prompt: %s", truncated_prompt)

            # Generate response
            try:
                async for token in llm_manager.generate_stream(prompt):
                    yield json.dumps({"status": None, "content": token})
                    await asyncio.sleep(0)
            except (RuntimeError, ValueError, OSError) as e:
                logging.error("生成エラー: %s", e, exc_info=True)
                yield json.dumps({"status": f"エラー: {e}", "content": ""})
                raise

            yield json.dumps({"status": "完了", "content": ""})
            await asyncio.sleep(0)
        except (RuntimeError, ValueError, OSError, KeyError) as e:
            logging.error("[CHAT] event_generator エラー: %s", e, exc_info=True)
            yield json.dumps({"status": f"エラー: {e}", "content": ""})
            await asyncio.sleep(0)

    return StreamingResponse(event_generator(), media_type="text/plain")


@app.post("/upload_epub")
def upload_epub(epub_file: UploadFile = File(...)) -> RedirectResponse:
    """Upload an EPUB file."""
    filename = epub_file.filename or "uploaded.epub"
    file_location = os.path.join(EPUB_DIR, filename)
    with open(file_location, "wb") as f:
        f.write(epub_file.file.read())
    book = epub.read_epub(file_location)
    title = book.get_metadata("DC", "title")
    if title and len(title) > 0:
        safe_title = title[0][0].strip().replace("/", "_")
        new_location = os.path.join(EPUB_DIR, safe_title + ".epub")
        if not os.path.exists(new_location):
            os.rename(file_location, new_location)
    return RedirectResponse(url="/bookshelf_ui", status_code=303)


@app.post("/delete_epub")
def delete_epub(data: dict[str, Any] = Body(...)) -> JSONResponse:
    """Delete an EPUB file and its associated data."""
    book_id: str | None = data.get("book_id")
    if not book_id:
        return JSONResponse({"error": "book_id required"}, status_code=400)

    result = enhanced_epub_service.delete_book(book_id)
    if "error" in result:
        return JSONResponse(result, status_code=500)
    return JSONResponse(result)


@app.get("/download_epub/{book_id}")
def download_epub(book_id: str) -> Any:
    """Download an EPUB file."""
    epub_path = os.path.join(EPUB_DIR, book_id)
    if not os.path.exists(epub_path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(epub_path, media_type="application/epub+zip", filename=book_id)


@app.get("/history/{session_id}", response_class=JSONResponse)
def history_detail(session_id: str) -> JSONResponse:
    """Get chat history for a specific session."""
    history = chat_service.get_session_history(session_id)
    if history is None:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return JSONResponse(history)


@app.get("/session/{session_id}", response_class=JSONResponse)
def session_detail(session_id: str) -> JSONResponse:
    """セッションの全データ（メッセージ + 書籍選択情報）を取得"""
    session_data = chat_service.get_session_data(session_id)
    if session_data is None:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return JSONResponse(session_data)


@app.post("/history/{session_id}", response_class=JSONResponse)
def post_history(session_id: str, data: dict[str, Any] = Body(...)) -> JSONResponse:
    """Save chat history for a specific session."""
    messages = data.get("messages", [])
    book_ids = data.get("book_ids", [])
    result = chat_service.save_session(session_id, messages, book_ids)
    if "error" in result:
        return JSONResponse(result, status_code=500)
    return JSONResponse(result)


@app.delete("/history/{session_id}", response_class=JSONResponse)
def delete_history_endpoint(session_id: str) -> JSONResponse:
    """セッションの履歴を削除"""
    result = chat_service.delete_session(session_id)
    if "error" in result:
        status_code = 404 if result["error"] == "Session not found" else 500
        return JSONResponse(result, status_code=status_code)
    return JSONResponse(result)
