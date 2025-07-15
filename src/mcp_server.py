"""FastMCP server integration for EPUB LLM.

This module provides FastMCP tool integrations for the EPUB LLM application,
exposing key functionality as tools for external consumption.
"""

# pylint: disable=duplicate-code

import asyncio
import json
import logging
import sys
from typing import Any, cast

from fastmcp import FastMCP

from src.app import chat_service, enhanced_epub_service, epub_service

# ログ設定
logger = logging.getLogger(__name__)

# FastMCP アプリケーション
mcp_app: FastMCP = FastMCP("epub-llm")


def validate_json_response(data: Any) -> Any:
    """JSONレスポンスを検証し、問題のある文字を修正する"""
    try:
        # まずJSONシリアライズを試行
        json.dumps(data, ensure_ascii=False)
        return data
    except (TypeError, ValueError) as e:
        logger.warning("JSON serialization failed: %s", e)

        # 文字列の場合、問題のある文字を修正
        if isinstance(data, str):
            # 制御文字や無効なUnicode文字を除去
            cleaned = "".join(
                char for char in data if char.isprintable() or char.isspace()
            )
            # 問題のある文字を安全な文字に置換
            cleaned = cleaned.replace("�", "?")
            return cleaned

        # リストの場合、各要素を再帰的に検証
        if isinstance(data, list):
            return [validate_json_response(item) for item in data]

        # 辞書の場合、各値を再帰的に検証
        if isinstance(data, dict):
            return {key: validate_json_response(value) for key, value in data.items()}

        # その他の型の場合、文字列に変換
        return str(data)


# FastMCP にはミドルウェア機能がないため、個別にエラーハンドリングを実装


@mcp_app.tool()
def list_epub_books() -> list[dict[str, Any]]:
    """EPUBファイル一覧を取得"""
    try:
        result = epub_service.get_bookshelf()
        logger.info("list_epub_books: %d冊の書籍を返却", len(result))
        return cast(list[dict[str, Any]], validate_json_response(result))
    except (ImportError, AttributeError, OSError, RuntimeError) as e:
        logger.error("list_epub_books エラー: %s", e, exc_info=True)
        return []


@mcp_app.tool()
def get_epub_metadata(book_id: str) -> dict[str, Any]:
    """指定されたEPUBのメタデータを取得"""
    try:
        result = epub_service.get_book_metadata(book_id)
        logger.info("get_epub_metadata: %s のメタデータを取得", book_id)
        return cast(dict[str, Any], validate_json_response(result))
    except (ImportError, AttributeError, OSError, RuntimeError, KeyError) as e:
        logger.error("get_epub_metadata エラー (%s): %s", book_id, e, exc_info=True)
        return cast(dict[str, Any], validate_json_response({"error": str(e)}))


@mcp_app.tool()
def search_epub_content(
    book_id: str, query: str, top_k: int = 5
) -> list[dict[str, Any]]:
    """指定されたEPUBから関連コンテンツを検索（スマートRAG使用）"""
    try:
        result = enhanced_epub_service.search_single_book(book_id, query, top_k)
        logger.info("search_epub_content: %s で '%s...' を検索", book_id, query[:50])
        return cast(list[dict[str, Any]], validate_json_response(result))
    except (
        ImportError,
        AttributeError,
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
    ) as e:
        logger.error(
            "search_epub_content エラー (%s, %s): %s", book_id, query, e, exc_info=True
        )
        return cast(list[dict[str, Any]], validate_json_response([{"error": str(e)}]))


@mcp_app.tool()
def get_context_for_books(book_ids: list[str], query: str, top_k: int = 10) -> str:
    """複数の書籍からクエリに関連するコンテキストを取得（スマートRAG使用）"""
    try:
        result = enhanced_epub_service.get_context_for_query(query, book_ids, top_k)
        logger.info(
            "get_context_for_books: %d冊の書籍で '%s...' を検索",
            len(book_ids),
            query[:50],
        )
        return cast(str, validate_json_response(result))
    except (
        ImportError,
        AttributeError,
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
    ) as e:
        logger.error(
            "get_context_for_books エラー (%s, %s): %s",
            book_ids,
            query,
            e,
            exc_info=True,
        )
        return cast(str, validate_json_response(f"エラー: {str(e)}"))


@mcp_app.tool()
def smart_search_books(book_ids: list[str], query: str, top_k: int = 10) -> str:
    """高度なハイブリッド検索でコンテキストを取得"""
    try:

        logger.info(
            "smart_search_books: %d冊の書籍で '%s...' をスマート検索",
            len(book_ids),
            query[:50],
        )

        # 非同期関数を同期的に実行するためのヘルパー関数
        def run_async_safe(coro: Any) -> Any:
            """非同期コルーチンを安全に実行"""
            try:
                # 既存のイベントループを取得を試行
                asyncio.get_running_loop()
                # 実行中のループがある場合は新しいスレッドで実行
                import concurrent.futures  # pylint: disable=import-outside-toplevel

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result(timeout=300)  # 5分のタイムアウト
            except RuntimeError:
                # イベントループが存在しない場合、直接実行
                try:
                    return asyncio.run(coro)
                except Exception as async_error:
                    logger.error("Async execution failed: %s", async_error)
                    raise
            except Exception as e:
                logger.error("Async safe execution failed: %s", e)
                raise

        result = run_async_safe(
            enhanced_epub_service.smart_search_books(book_ids, query, top_k)
        )
        return cast(str, validate_json_response(result))
    except (
        ImportError,
        AttributeError,
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
    ) as e:
        logger.error(
            "smart_search_books エラー (%s, %s): %s", book_ids, query, e, exc_info=True
        )
        return cast(str, validate_json_response(f"エラー: {str(e)}"))


@mcp_app.tool()
def get_chat_histories() -> list[str]:
    """チャット履歴一覧を取得"""
    try:
        sessions = chat_service.get_all_sessions()
        session_ids = [
            session.get("session_id", "")
            for session in sessions
            if session.get("session_id")
        ]
        logger.info("get_chat_histories: %d件の履歴を返却", len(session_ids))
        return cast(list[str], validate_json_response(session_ids))
    except (
        ImportError,
        AttributeError,
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
    ) as e:
        logger.error("get_chat_histories エラー: %s", e, exc_info=True)
        return []


@mcp_app.tool()
def get_chat_history(session_id: str) -> list[dict[str, Any]]:
    """指定されたセッションのチャット履歴を取得"""
    try:
        history = chat_service.get_session_history(session_id)
        result = history if history is not None else []
        logger.info(
            "get_chat_history: %s の %d件のメッセージを返却", session_id, len(result)
        )
        return cast(list[dict[str, Any]], validate_json_response(result))
    except (
        ImportError,
        AttributeError,
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
    ) as e:
        logger.error("get_chat_history エラー (%s): %s", session_id, e, exc_info=True)
        return []


if __name__ == "__main__":
    # MCP server用のエントリーポイント
    logger.info("MCP Server を直接起動中...")
    try:
        mcp_app.run()
    except (
        ImportError,
        AttributeError,
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
    ) as e:
        logger.error("MCP Server 直接起動エラー: %s", e, exc_info=True)
        sys.exit(1)
