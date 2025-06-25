import asyncio
import json
import logging
import os
import threading
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import mlx_lm
import numpy as np
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
from fastmcp import FastMCPServer
from mlx_embeddings import load as load_embedding_model
from mlx_lm.sample_utils import make_sampler

from src.embedding_util import (
    build_faiss_index,
    create_embeddings_from_texts,
    load_embeddings,
    save_embeddings,
    search_similar,
)
from src.epub_util import extract_epub_metadata, extract_epub_text, get_epub_cover_path

from .history_util import list_histories, load_history, save_history

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

EPUB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../epub"))
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../cache"))
HISTORY_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../cache/history")
)
os.makedirs(HISTORY_DIR, exist_ok=True)

MODEL, TOKENIZER = mlx_lm.load("lmstudio-community/Llama-4-Scout-17B-16E-MLX-text-4bit")

LOG_FILE = os.path.join(os.path.dirname(__file__), "../epub-llm.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

EMBED_MODEL, EMBED_TOKENIZER = load_embedding_model(
    "mlx-community/multilingual-e5-base-mlx"
)


def save_history(session_id: str, history: List[Dict[str, Union[str, None]]]) -> None:
    path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def load_history(
    session_id: str,
) -> Optional[List[Dict[str, Union[str, None]]]]:
    path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_histories() -> List[str]:
    return [f[:-5] for f in os.listdir(HISTORY_DIR) if f.endswith(".json")]


@app.get("/", response_class=HTMLResponse)
def chat_ui(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/bookshelf", response_class=JSONResponse)
def bookshelf() -> list[dict[str, str | None]]:
    books = []
    for fname in os.listdir(EPUB_DIR):
        if fname.lower().endswith(".epub"):
            epub_path = os.path.join(EPUB_DIR, fname)
            meta = extract_epub_metadata(epub_path)
            title = meta.get("title") or fname
            cover_url = None
            cache_dir = os.path.join(os.path.dirname(__file__), "../static/cache")
            cover_path = get_epub_cover_path(epub_path, cache_dir)
            if cover_path:
                cover_url = "/static/cache/" + os.path.basename(cover_path)
            books.append(
                {
                    "id": fname,
                    "title": title,
                    "cover": cover_url,
                    "author": meta.get("author"),
                    "year": meta.get("year"),
                }
            )
    return books


@app.get("/bookshelf_ui", response_class=HTMLResponse)
def bookshelf_ui(request: Request) -> HTMLResponse:
    books = bookshelf()
    return templates.TemplateResponse(
        "bookshelf.html", {"request": request, "books": books}
    )


@app.post("/chat")
async def chat_stream(request: Request) -> StreamingResponse:
    body = await request.json()

    async def event_generator() -> AsyncGenerator[str, None]:
        book_ids: list[str] = body.get("book_ids", [])
        messages: list[dict[str, str]] = body.get("messages", [])

        system_msg: dict[str, str] = {
            "role": "system",
            "content": """
あなたは日本語のMarkdownで答えるアシスタントAIです。
""",
        }
        messages.insert(0, system_msg)
        logging.info("[embedding] created for %s", book_ids)
        if book_ids:
            all_embeddings = []
            all_texts = []
            all_book_ids = []
            for book_id in book_ids:
                base_path = os.path.join(CACHE_DIR, book_id)
                try:
                    if not (
                        os.path.exists(base_path + ".npy")
                        and os.path.exists(base_path + ".json")
                    ):
                        epub_path = os.path.join(EPUB_DIR, book_id)
                        text = extract_epub_text(epub_path, base_path + ".txt")
                        chunk_size = 4000
                        overlap = 500
                        text_chunks = []
                        text_len = len(text)
                        start = 0
                        while start < text_len:
                            end = min(start + chunk_size, text_len)
                            text_chunks.append(text[start:end])
                            if end == text_len:
                                break
                            start += chunk_size - overlap
                        embeddings = create_embeddings_from_texts(
                            text_chunks, EMBED_MODEL, EMBED_TOKENIZER
                        )
                        save_embeddings(embeddings, text_chunks, base_path)
                        logging.info("[embedding] created for %s", book_id)
                    embeddings, texts = load_embeddings(base_path)
                    all_embeddings.append(embeddings)
                    all_texts.extend(texts)
                    all_book_ids.extend([book_id] * len(texts))
                except Exception as e:
                    err_msg = f"[embedding] failed to load for {book_id}: {e}"
                    logging.warning(err_msg)
                    yield json.dumps(
                        {
                            "status": err_msg,
                            "content": "",
                        }
                    )
                    await asyncio.sleep(0)
            if all_embeddings:
                book_id_to_title = {}
                for book_id in book_ids:
                    epub_path = os.path.join(EPUB_DIR, book_id)
                    meta = extract_epub_metadata(epub_path)
                    title = meta.get("title") or book_id
                    book_id_to_title[book_id] = title
                embeddings = np.concatenate(all_embeddings, axis=0)
                texts = all_texts
                book_ids_for_chunks = all_book_ids
                index = build_faiss_index(embeddings)

                # RAG用クエリ生成とコンテキスト取得
                context, results = get_rag_context(
                    messages=messages,
                    model=EMBED_MODEL,
                    tokenizer=EMBED_TOKENIZER,
                    index=index,
                    texts=texts,
                    top_k=10,
                )
                # systemメッセージにコンテキスト追加
                messages[0]["content"] += f"\n\n## コンテキスト\n{context}\n"
        yield json.dumps(
            {
                "status": f"[LLM] 生成開始... context_size: {len(messages[0]['content'])}",
                "content": "",
            }
        )
        await asyncio.sleep(0)
        logging.info("[LLM] 生成開始...")

        # LLMで最終回答生成
        prompt = TOKENIZER.apply_chat_template(messages, add_generation_prompt=True)
        token_count: int = 0
        start_time: Optional[float] = None
        for token in mlx_lm.stream_generate(
            MODEL,
            TOKENIZER,
            prompt=prompt,
            max_tokens=128000,
            sampler=make_sampler(temp=0.2, top_p=1.0, top_k=40),
        ):
            if token.text:
                if start_time is None:
                    start_time = time.time()
                    logging.info("[LLM] 1st token generated.")
                token_count += 1
                yield json.dumps({"status": "[LLM] 生成中...", "content": token.text})
                await asyncio.sleep(0)

        if start_time is not None:
            elapsed: float = time.time() - start_time
            tps: float = token_count / elapsed if elapsed > 0 else 0.0
            logging.info(
                "[LLM] generate complete. token_count=%d elapsed=%.2fs tps=%.2f",
                token_count,
                elapsed,
                tps,
            )

        yield json.dumps(
            {
                "status": (
                    f"[LLM] 生成完了 token_count={token_count} "
                    f"elapsed={elapsed:.2f}s tps={tps:.2f}"
                ),
                "content": "",
            }
        )
        await asyncio.sleep(0)
        logging.info("LLM 生成完了")

    return StreamingResponse(event_generator(), media_type="text/plain")


@app.post("/upload_epub")
def upload_epub(epub_file: UploadFile = File(...)) -> RedirectResponse:
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
    book_id: str | None = data.get("book_id")
    if not book_id:
        return JSONResponse({"error": "book_id required"}, status_code=400)
    epub_path = os.path.join(EPUB_DIR, book_id)
    try:
        if os.path.exists(epub_path):
            os.remove(epub_path)
        cover_path = os.path.join(
            os.path.dirname(__file__), "../static/cache", book_id + ".cover.jpg"
        )
        if os.path.exists(cover_path):
            os.remove(cover_path)
        cache_path = os.path.join(CACHE_DIR, book_id + ".txt")
        if os.path.exists(cache_path):
            os.remove(cache_path)
        return JSONResponse({"result": "ok"})
    except (OSError, ValueError) as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/download_epub/{book_id}")
def download_epub(book_id: str) -> Any:
    epub_path = os.path.join(EPUB_DIR, book_id)
    if not os.path.exists(epub_path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(epub_path, media_type="application/epub+zip", filename=book_id)


# --- 履歴ユーティリティ関数を history_util.py へ移動 ---
# 実行結果をファイル出力し、進捗・完了を管理するAPI
output_status: Dict[str, Dict[str, Union[str, float, int]]] = {}


def background_output(session_id: str, content: str) -> None:
    path = os.path.join(HISTORY_DIR, f"{session_id}.txt")
    output_status[session_id] = {"status": "running", "progress": 0}
    try:
        chunks = [content[i : i + 4096] for i in range(0, len(content), 4096)]
        for i, chunk in enumerate(chunks):
            if i > 0:
                f = open(path, "a", encoding="utf-8")
            else:
                f = open(path, "w", encoding="utf-8")
            f.write(chunk)
            f.flush()
            f.close()
            output_status[session_id]["progress"] = (i + 1) * 4096 / len(content)
            time.sleep(0.05)
        output_status[session_id]["status"] = "done"
        output_status[session_id]["progress"] = 1.0
    except OSError as e:
        output_status[session_id]["status"] = f"error: {e}"
    except ValueError as e:
        output_status[session_id]["status"] = f"error: {e}"


@app.post("/output_file/{session_id}", response_class=JSONResponse)
def output_file(session_id: str, data: Dict[str, str] = Body(...)) -> Dict[str, str]:
    content = data.get("content", "")
    t = threading.Thread(target=background_output, args=(session_id, content))
    t.start()
    return {"result": "started"}


@app.get("/output_status/{session_id}", response_class=JSONResponse)
def output_status_api(session_id: str) -> Dict[str, Union[str, float]]:
    return output_status.get(session_id, {"status": "not_found", "progress": 0})


@app.get("/output_file/{session_id}", response_class=FileResponse)
def get_output_file(session_id: str) -> Any:
    path = os.path.join(HISTORY_DIR, f"{session_id}.txt")
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="text/plain", filename=f"{session_id}.txt")


@app.get("/history_list", response_class=JSONResponse)
def history_list() -> List[str]:
    return list_histories()


@app.get("/history/{session_id}", response_class=JSONResponse)
def history_detail(session_id: str) -> Any:
    history = load_history(session_id)
    if history is None:
        return JSONResponse([], status_code=404)
    return history


@app.post("/history/{session_id}", response_class=JSONResponse)
def post_history(
    session_id: str,
    data: List[Dict[str, Union[str, None]]] = Body(...),
) -> Dict[str, str]:
    save_history(session_id, data)
    return {"result": "ok"}


# FastMCPServerを /mcp にマウント
mcp_server = FastMCPServer(app, "/mcp")


def get_rag_query_from_llm(
    messages: List[Dict[str, str]],
    model: Any,
    tokenizer: Any,
    max_tokens: int = 128,
) -> str:
    """
    LLMに問い合わせてRAG用クエリ文を生成する。
    """
    prompt_messages = list(messages)
    prompt_messages.append(
        {
            "role": "user",
            "content": "RAG検索用に意図抽出したクエリ文を1文だけ日本語で出力してください。",
        }
    )
    prompt = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True)
    llm_query = ""
    for token in mlx_lm.stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=0.2, top_p=1.0, top_k=40),
    ):
        if token.text:
            llm_query += token.text
    return llm_query.strip()


def get_rag_context(
    messages: List[Dict[str, str]],
    model: Any,
    tokenizer: Any,
    index: Any,
    texts: List[str],
    top_k: int = 10,
) -> Tuple[str, List[Tuple[int, float, str]]]:
    """
    LLMでRAG用クエリを生成し、そのクエリでembedding検索してcontextを返す。
    """
    rag_query = get_rag_query_from_llm(messages, model, tokenizer)
    logging.info("[RAG] query: %s", rag_query)
    results = search_similar(
        query=rag_query,
        model=model,
        tokenizer=tokenizer,
        index=index,
        texts=texts,
        top_k=top_k,
    )
    context = "\n---\n".join([r[2] for r in results])
    return context, results


def generate_llm_response(
    messages: List[Dict[str, str]],
    model: Any,
    tokenizer: Any,
    max_tokens: int = 128000,
) -> str:
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    return mlx_lm.generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=0.2, top_p=1.0, top_k=40),
    )
