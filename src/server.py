"""FastMCP server integration for EPUB LLM.

This module provides FastMCP tool integrations for the EPUB LLM application,
exposing key functionality as tools for external consumption.
"""

import asyncio
from typing import Any

from fastmcp import FastMCP

# FastMCP アプリケーション
mcp_app: FastMCP = FastMCP("/mcp")


@mcp_app.tool()
def list_epub_books() -> list[dict[str, Any]]:
    """EPUBファイル一覧を取得"""
    from src.app import epub_service  # pylint: disable=import-outside-toplevel

    return epub_service.get_bookshelf()


@mcp_app.tool()
def get_epub_metadata(book_id: str) -> dict[str, Any]:
    """指定されたEPUBのメタデータを取得"""
    from src.app import epub_service  # pylint: disable=import-outside-toplevel

    return epub_service.get_book_metadata(book_id)


@mcp_app.tool()
def search_epub_content(
    book_id: str, query: str, top_k: int = 5
) -> list[dict[str, Any]]:
    """指定されたEPUBから関連コンテンツを検索（スマートRAG使用）"""
    from src.app import enhanced_epub_service  # pylint: disable=import-outside-toplevel

    return enhanced_epub_service.search_single_book(book_id, query, top_k)


@mcp_app.tool()
def get_context_for_books(book_ids: list[str], query: str, top_k: int = 10) -> str:
    """複数の書籍からクエリに関連するコンテキストを取得（スマートRAG使用）"""
    from src.app import enhanced_epub_service  # pylint: disable=import-outside-toplevel

    return enhanced_epub_service.get_context_for_query(query, book_ids, top_k)


@mcp_app.tool()
def smart_search_books(book_ids: list[str], query: str, top_k: int = 10) -> str:
    """高度なハイブリッド検索でコンテキストを取得"""
    from src.app import enhanced_epub_service  # pylint: disable=import-outside-toplevel

    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            enhanced_epub_service.smart_search_books(book_ids, query, top_k)
        )
    except RuntimeError:
        return asyncio.run(
            enhanced_epub_service.smart_search_books(book_ids, query, top_k)
        )


@mcp_app.tool()
def get_chat_histories() -> list[str]:
    """チャット履歴一覧を取得"""
    from src.app import chat_service  # pylint: disable=import-outside-toplevel

    sessions = chat_service.get_all_sessions()
    return [
        session.get("session_id", "")
        for session in sessions
        if session.get("session_id")
    ]


@mcp_app.tool()
def get_chat_history(session_id: str) -> list[dict[str, Any]]:
    """指定されたセッションのチャット履歴を取得"""
    from src.app import chat_service  # pylint: disable=import-outside-toplevel

    history = chat_service.get_session_history(session_id)
    return history if history is not None else []


if __name__ == "__main__":
    # MCP server用のエントリーポイント
    mcp_app.run()
