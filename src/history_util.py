"""Chat history management utilities.

Provides saving/loading of chat sessions with simple metadata so that the
web app and tests can persist and query conversation history.
Supports both a legacy list-only format and a structured format.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

HISTORY_DIR = os.path.join(os.path.dirname(__file__), "../cache/history")

LOGGER = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds")


def ensure_history_dir() -> None:
    """Ensure history directory exists."""
    os.makedirs(HISTORY_DIR, exist_ok=True)


def _session_path(session_id: str) -> str:
    return os.path.join(HISTORY_DIR, f"{session_id}.json")


def save_history(
    session_id: str,
    messages: list[dict[str, Any]],
    book_ids: list[str] | None = None,
) -> None:
    """Save chat history in a structured JSON shape.

    Adds timestamps to individual messages when missing and maintains
    created_at/updated_at fields on the session file.
    """
    ensure_history_dir()
    path = _session_path(session_id)

    # Normalize messages: ensure timestamp exists for each message
    normalized: list[dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        item = dict(m)
        item.setdefault("timestamp", _now_iso())
        normalized.append(item)

    now = _now_iso()
    data: dict[str, Any]
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict) and "messages" in existing:
                data = existing
                data["messages"] = normalized
                data["book_ids"] = list(book_ids or data.get("book_ids") or [])
                data["updated_at"] = now
            else:
                # legacy -> overwrite to new format
                data = {
                    "messages": normalized,
                    "book_ids": list(book_ids or []),
                    "created_at": now,
                    "updated_at": now,
                }
        except (OSError, ValueError, TypeError) as exc:  # rewrite if corrupted
            LOGGER.debug("Rewriting corrupted history file %s: %s", path, exc)
            data = {
                "messages": normalized,
                "book_ids": list(book_ids or []),
                "created_at": now,
                "updated_at": now,
            }
    else:
        data = {
            "messages": normalized,
            "book_ids": list(book_ids or []),
            "created_at": now,
            "updated_at": now,
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_history(session_id: str) -> list[dict[str, Any]] | None:
    """Load messages list for a session.

    Returns legacy list or structured messages list; None if missing/corrupted.
    """
    path = _session_path(session_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError, TypeError):
        return None

    if isinstance(data, list):
        return [m for m in data if isinstance(m, dict)]
    if isinstance(data, dict):
        msgs = data.get("messages")
        if isinstance(msgs, list):
            return [m for m in msgs if isinstance(m, dict)]
    return None


def load_session_data(session_id: str) -> dict[str, Any] | None:
    """Load structured session data, converting legacy format if necessary."""
    path = _session_path(session_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError, TypeError):
        return None

    if isinstance(data, dict) and "messages" in data:
        # Ensure minimal keys exist
        data.setdefault("book_ids", [])
        data.setdefault("created_at", _now_iso())
        data.setdefault("updated_at", _now_iso())
        return data
    if isinstance(data, list):
        now = _now_iso()
        return {
            "messages": [m for m in data if isinstance(m, dict)],
            "book_ids": [],
            "created_at": now,
            "updated_at": now,
        }
    return None


def get_all_sessions() -> list[str]:
    """Return list of session IDs found in the history dir."""
    if not os.path.isdir(HISTORY_DIR):
        return []
    out: list[str] = []
    for name in os.listdir(HISTORY_DIR):
        if name.endswith(".json"):
            out.append(os.path.splitext(name)[0])
    return sorted(out)


def delete_history(session_id: str) -> bool:
    """Delete a session file if present; return True if removed."""
    path = _session_path(session_id)
    if not os.path.exists(path):
        return False
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def get_session_summary(session_id: str) -> dict[str, Any] | None:
    """Return a compact summary for a session, or None if missing."""
    data = load_session_data(session_id)
    if not data:
        return None
    messages = data.get("messages", []) if isinstance(data, dict) else []
    # First user message content (truncate to 100 chars)
    first_user = next((m for m in messages if m.get("role") == "user"), None)
    first_content: str | None = None
    if isinstance(first_user, dict):
        c = str(first_user.get("content", ""))
        if len(c) > 100:
            first_content = c[:100] + "..."
        else:
            first_content = c

    return {
        "session_id": session_id,
        "message_count": len(messages),
        "first_message": first_content,
        "last_updated": data.get("updated_at"),
    }
