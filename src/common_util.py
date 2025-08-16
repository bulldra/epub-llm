"""
共通ユーティリティモジュール

重複コードを削減し、コードの再利用性を向上させるための共通関数群
"""

import logging
import os
from typing import Any

from src.epub_util import extract_epub_metadata, extract_epub_toc, get_epub_cover_path


def get_book_list(epub_dir: str) -> list[dict[str, Any]]:
    """共通のブック一覧取得処理"""
    books = []
    for fname in os.listdir(epub_dir):
        if fname.lower().endswith(".epub"):
            epub_path = os.path.join(epub_dir, fname)
            meta = extract_epub_metadata(epub_path)
            toc = extract_epub_toc(epub_path)
            title = meta.get("title") or fname
            author = meta.get("author") or ""

            cover_url = None
            cache_dir = os.path.join(os.path.dirname(__file__), "../static/cache")
            cover_path = get_epub_cover_path(epub_path, cache_dir)
            if cover_path:
                cover_url = "/static/cache/" + os.path.basename(cover_path)

            # Convert TOC to simple title list for compatibility
            toc_titles = [item["title"] for item in toc] if toc else []

            books.append(
                {
                    "id": fname,
                    "title": title,
                    "cover": cover_url,
                    "author": author,
                    "year": meta.get("year"),
                    "toc": toc_titles,
                }
            )
    return books


def get_book_title_from_metadata(epub_dir: str, book_id: str) -> str:
    """共通のブックタイトル取得処理"""
    epub_path = os.path.join(epub_dir, book_id)
    try:
        meta = extract_epub_metadata(epub_path)
        return meta.get("title", book_id)
    except (OSError, ValueError, KeyError):
        return book_id


def delete_book_files(epub_dir: str, cache_dir: str, book_id: str) -> None:
    """共通のブックファイル削除処理"""
    epub_path = os.path.join(epub_dir, book_id)

    # Delete EPUB file
    if os.path.exists(epub_path):
        os.remove(epub_path)

    # Delete cover cache
    cover_path = os.path.join(
        os.path.dirname(__file__), "../static/cache", book_id + ".cover.jpg"
    )
    if os.path.exists(cover_path):
        os.remove(cover_path)

    # Delete text cache
    cache_path = os.path.join(cache_dir, book_id + ".txt")
    if os.path.exists(cache_path):
        os.remove(cache_path)

    # Delete embeddings and BM25 index
    files_to_delete = [
        os.path.join(cache_dir, book_id + ".npy"),
        os.path.join(cache_dir, book_id + ".json"),
        os.path.join(cache_dir, book_id + ".bm25.json"),
    ]
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)


def setup_common_logger(name: str) -> logging.Logger:
    """共通のロガー設定"""
    return logging.getLogger(name)


def create_text_chunks(
    text: str, chunk_size: int = 1000, overlap: int = 200
) -> list[str]:
    """共通のテキストチャンク作成処理"""
    chunks = []
    text_len = len(text)
    start = 0

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to end at sentence boundary
        if end < text_len:
            for punct in ["。", "！", "？", ".", "!", "?"]:
                punct_pos = text.rfind(punct, start, end)
                if punct_pos > start + chunk_size // 2:
                    end = punct_pos + 1
                    break

        chunks.append(text[start:end])
        start = max(start + chunk_size - overlap, end)

    return chunks


def format_chat_response(
    prompt: str, context_size: int, messages: list[dict[str, str]]
) -> dict[str, Any]:
    """共通のチャットレスポンス形式作成"""
    return {
        "prompt": prompt,
        "context_size": context_size,
        "messages": messages,
    }
