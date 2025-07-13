"""Chat history management utilities.

This module provides functionality for managing chat session history,
including loading, saving, and managing chat conversations with metadata.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any

HISTORY_DIR = os.path.join(os.path.dirname(__file__), "../cache/history")


def ensure_history_dir() -> None:
    """履歴ディレクトリを作成（存在しない場合）"""
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR, exist_ok=True)
        logging.info("Created history directory: %s", HISTORY_DIR)


def save_history(
    session_id: str,
    history: list[dict[str, str | None]],
    book_ids: list[str] | None = None,
) -> None:
    """セッションの履歴を保存（書籍選択情報も含む）"""
    ensure_history_dir()

    # タイムスタンプを追加
    for message in history:
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()

    # 保存データの構造
    save_data = {
        "messages": history,
        "book_ids": book_ids or [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        logging.info(
            "Saved history for session: %s with %d books",
            session_id,
            len(book_ids or []),
        )
    except (OSError, TypeError, ValueError) as e:
        logging.error("Failed to save history for session %s: %s", session_id, e)
        raise


def load_history(session_id: str) -> list[dict[str, Any]] | None:
    """セッションの履歴を読み込み（メッセージのみ返却、後方互換性あり）"""
    ensure_history_dir()

    path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        logging.info("No history found for session: %s", session_id)
        return None

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # 新形式（辞書）か旧形式（リスト）かを判定
        history: list[dict[str, Any]]
        if isinstance(data, dict) and "messages" in data:
            # 新形式
            history = data["messages"]
            logging.info(
                "Loaded history for session: %s (new format with %d books)",
                session_id,
                len(data.get("book_ids", [])),
            )
        else:
            # 旧形式（後方互換性）
            history = data
            logging.info("Loaded history for session: %s (legacy format)", session_id)

        return history
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
        logging.error("Failed to load history for session %s: %s", session_id, e)
        return None


def load_session_data(session_id: str) -> dict[str, Any] | None:
    """セッションの全データを読み込み（メッセージ + 書籍選択情報）"""
    ensure_history_dir()

    path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        logging.info("No session data found for session: %s", session_id)
        return None

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # 新形式（辞書）か旧形式（リスト）かを判定
        if isinstance(data, dict) and "messages" in data:
            # 新形式
            logging.info("Loaded session data for: %s (new format)", session_id)
            return data

        # 旧形式を新形式に変換
        logging.info("Loaded session data for: %s (converted from legacy)", session_id)
        return {
            "messages": data,
            "book_ids": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
        logging.error("Failed to load session data for %s: %s", session_id, e)
        return None


def get_all_sessions() -> list[str]:
    """すべてのセッションIDを取得"""
    ensure_history_dir()

    try:
        sessions = []
        for filename in os.listdir(HISTORY_DIR):
            if filename.endswith(".json"):
                sessions.append(filename[:-5])  # .jsonを除去
        return sorted(sessions, reverse=True)  # 新しい順
    except OSError as e:
        logging.error("Failed to get sessions: %s", e)
        return []


def delete_history(session_id: str) -> bool:
    """セッションの履歴を削除"""
    ensure_history_dir()

    path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        return False

    try:
        os.remove(path)
        logging.info("Deleted history for session: %s", session_id)
        return True
    except OSError as e:
        logging.error("Failed to delete history for session %s: %s", session_id, e)
        return False


def get_session_summary(session_id: str) -> dict[str, Any] | None:
    """セッションの概要を取得"""
    history = load_history(session_id)
    if not history:
        return None

    # 最初のユーザーメッセージを取得
    first_user_message = None
    for message in history:
        if message.get("role") == "user":
            first_user_message = message.get("content", "")
            break

    # 概要を作成
    summary = {
        "session_id": session_id,
        "message_count": len(history),
        "first_message": (
            first_user_message[:100] + "..."
            if first_user_message and len(first_user_message) > 100
            else first_user_message
        ),
        "last_updated": history[-1].get("timestamp") if history else None,
    }

    return summary
