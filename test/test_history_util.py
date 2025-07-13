"""Tests for history_util module."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from src.history_util import (
    delete_history,
    ensure_history_dir,
    get_all_sessions,
    get_session_summary,
    load_history,
    load_session_data,
    save_history,
)


class TestHistoryUtil:
    """Test cases for history_util functions."""

    def test_ensure_history_dir_creates_directory(self):
        """Test that ensure_history_dir creates the directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_dir = os.path.join(temp_dir, "history")

            with patch("src.history_util.HISTORY_DIR", history_dir):
                ensure_history_dir()
                assert os.path.exists(history_dir)

    def test_ensure_history_dir_existing_directory(self):
        """Test that ensure_history_dir works with existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_dir = os.path.join(temp_dir, "history")
            os.makedirs(history_dir)

            with patch("src.history_util.HISTORY_DIR", history_dir):
                ensure_history_dir()  # Should not raise error
                assert os.path.exists(history_dir)

    def test_save_history_basic(self):
        """Test basic history saving functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_dir = os.path.join(temp_dir, "history")
            session_id = "test_session"
            history = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
            book_ids = ["book1", "book2"]

            with patch("src.history_util.HISTORY_DIR", history_dir):
                save_history(session_id, history, book_ids)

                # Check file exists
                file_path = os.path.join(history_dir, f"{session_id}.json")
                assert os.path.exists(file_path)

                # Check file content
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)

                assert "messages" in data
                assert "book_ids" in data
                assert "created_at" in data
                assert "updated_at" in data
                assert data["book_ids"] == book_ids
                assert len(data["messages"]) == 2
                assert all("timestamp" in msg for msg in data["messages"])

    def test_save_history_without_book_ids(self):
        """Test saving history without book IDs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_dir = os.path.join(temp_dir, "history")
            session_id = "test_session"
            history = [{"role": "user", "content": "Hello"}]

            with patch("src.history_util.HISTORY_DIR", history_dir):
                save_history(session_id, history)

                file_path = os.path.join(history_dir, f"{session_id}.json")
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)

                assert data["book_ids"] == []

    def test_save_history_with_error(self):
        """Test save_history handles errors appropriately."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_dir = os.path.join(temp_dir, "history")
            session_id = "test_session"
            history = [{"role": "user", "content": "Hello"}]

            with patch("src.history_util.HISTORY_DIR", history_dir):
                # Mock open to raise an error
                with patch("builtins.open", side_effect=OSError("Permission denied")):
                    with pytest.raises(OSError):
                        save_history(session_id, history)

    def test_load_history_new_format(self):
        """Test loading history in new format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_dir = os.path.join(temp_dir, "history")
            os.makedirs(history_dir)
            session_id = "test_session"

            # Create test data in new format
            test_data = {
                "messages": [
                    {"role": "user", "content": "Hello", "timestamp": "2024-01-01T10:00:00"},
                    {"role": "assistant", "content": "Hi!", "timestamp": "2024-01-01T10:01:00"}
                ],
                "book_ids": ["book1"],
                "created_at": "2024-01-01T10:00:00",
                "updated_at": "2024-01-01T10:01:00"
            }

            file_path = os.path.join(history_dir, f"{session_id}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(test_data, f)

            with patch("src.history_util.HISTORY_DIR", history_dir):
                result = load_history(session_id)

                assert result == test_data["messages"]

    def test_load_history_legacy_format(self):
        """Test loading history in legacy format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_dir = os.path.join(temp_dir, "history")
            os.makedirs(history_dir)
            session_id = "test_session"

            # Create test data in legacy format (direct list)
            test_data = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"}
            ]

            file_path = os.path.join(history_dir, f"{session_id}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(test_data, f)

            with patch("src.history_util.HISTORY_DIR", history_dir):
                result = load_history(session_id)

                assert result == test_data

    def test_load_history_nonexistent(self):
        """Test loading history for nonexistent session."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_dir = os.path.join(temp_dir, "history")

            with patch("src.history_util.HISTORY_DIR", history_dir):
                result = load_history("nonexistent_session")

                assert result is None

    def test_load_history_corrupted_file(self):
        """Test loading history with corrupted JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_dir = os.path.join(temp_dir, "history")
            os.makedirs(history_dir)
            session_id = "test_session"

            # Create corrupted JSON file
            file_path = os.path.join(history_dir, f"{session_id}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("invalid json content")

            with patch("src.history_util.HISTORY_DIR", history_dir):
                result = load_history(session_id)

                assert result is None

    def test_load_session_data_new_format(self):
        """Test loading session data in new format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_dir = os.path.join(temp_dir, "history")
            os.makedirs(history_dir)
            session_id = "test_session"

            test_data = {
                "messages": [{"role": "user", "content": "Hello"}],
                "book_ids": ["book1"],
                "created_at": "2024-01-01T10:00:00",
                "updated_at": "2024-01-01T10:01:00"
            }

            file_path = os.path.join(history_dir, f"{session_id}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(test_data, f)

            with patch("src.history_util.HISTORY_DIR", history_dir):
                result = load_session_data(session_id)

                assert result == test_data

    def test_load_session_data_legacy_format(self):
        """Test loading session data in legacy format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_dir = os.path.join(temp_dir, "history")
            os.makedirs(history_dir)
            session_id = "test_session"

            # Legacy format (direct list)
            test_data = [{"role": "user", "content": "Hello"}]

            file_path = os.path.join(history_dir, f"{session_id}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(test_data, f)

            with patch("src.history_util.HISTORY_DIR", history_dir):
                result = load_session_data(session_id)

                assert result["messages"] == test_data
                assert result["book_ids"] == []
                assert "created_at" in result
                assert "updated_at" in result

    def test_get_all_sessions(self):
        """Test getting all session IDs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_dir = os.path.join(temp_dir, "history")
            os.makedirs(history_dir)

            # Create test session files
            sessions = ["session1", "session2", "session3"]
            for session in sessions:
                file_path = os.path.join(history_dir, f"{session}.json")
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump([], f)

            # Create a non-JSON file (should be ignored)
            with open(os.path.join(history_dir, "not_json.txt"), "w") as f:
                f.write("test")

            with patch("src.history_util.HISTORY_DIR", history_dir):
                result = get_all_sessions()

                assert len(result) == 3
                assert all(session in result for session in sessions)

    def test_get_all_sessions_empty_directory(self):
        """Test getting all sessions from empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_dir = os.path.join(temp_dir, "history")

            with patch("src.history_util.HISTORY_DIR", history_dir):
                result = get_all_sessions()

                assert result == []

    def test_delete_history_success(self):
        """Test successful history deletion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_dir = os.path.join(temp_dir, "history")
            os.makedirs(history_dir)
            session_id = "test_session"

            # Create test file
            file_path = os.path.join(history_dir, f"{session_id}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump([], f)

            with patch("src.history_util.HISTORY_DIR", history_dir):
                result = delete_history(session_id)

                assert result is True
                assert not os.path.exists(file_path)

    def test_delete_history_nonexistent(self):
        """Test deleting nonexistent history."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_dir = os.path.join(temp_dir, "history")

            with patch("src.history_util.HISTORY_DIR", history_dir):
                result = delete_history("nonexistent_session")

                assert result is False

    def test_get_session_summary_success(self):
        """Test getting session summary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_dir = os.path.join(temp_dir, "history")
            os.makedirs(history_dir)
            session_id = "test_session"

            # Create test data
            test_data = {
                "messages": [
                    {"role": "user", "content": "Hello world", "timestamp": "2024-01-01T10:00:00"},
                    {"role": "assistant", "content": "Hi!", "timestamp": "2024-01-01T10:01:00"}
                ],
                "book_ids": ["book1"],
                "created_at": "2024-01-01T10:00:00",
                "updated_at": "2024-01-01T10:01:00"
            }

            file_path = os.path.join(history_dir, f"{session_id}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(test_data, f)

            with patch("src.history_util.HISTORY_DIR", history_dir):
                result = get_session_summary(session_id)

                assert result["session_id"] == session_id
                assert result["message_count"] == 2
                assert result["first_message"] == "Hello world"
                assert result["last_updated"] == "2024-01-01T10:01:00"

    def test_get_session_summary_long_message(self):
        """Test getting session summary with long first message."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_dir = os.path.join(temp_dir, "history")
            os.makedirs(history_dir)
            session_id = "test_session"

            long_message = "A" * 150  # 150 characters
            test_data = {
                "messages": [
                    {"role": "user", "content": long_message, "timestamp": "2024-01-01T10:00:00"}
                ],
                "book_ids": [],
                "created_at": "2024-01-01T10:00:00",
                "updated_at": "2024-01-01T10:01:00"
            }

            file_path = os.path.join(history_dir, f"{session_id}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(test_data, f)

            with patch("src.history_util.HISTORY_DIR", history_dir):
                result = get_session_summary(session_id)

                assert result["first_message"] == "A" * 100 + "..."
                assert len(result["first_message"]) == 103  # 100 + "..."

    def test_get_session_summary_nonexistent(self):
        """Test getting summary for nonexistent session."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_dir = os.path.join(temp_dir, "history")

            with patch("src.history_util.HISTORY_DIR", history_dir):
                result = get_session_summary("nonexistent_session")

                assert result is None

    def test_get_session_summary_no_user_messages(self):
        """Test getting summary when no user messages exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_dir = os.path.join(temp_dir, "history")
            os.makedirs(history_dir)
            session_id = "test_session"

            test_data = {
                "messages": [
                    {"role": "assistant", "content": "Hello!", "timestamp": "2024-01-01T10:00:00"}
                ],
                "book_ids": [],
                "created_at": "2024-01-01T10:00:00",
                "updated_at": "2024-01-01T10:01:00"
            }

            file_path = os.path.join(history_dir, f"{session_id}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(test_data, f)

            with patch("src.history_util.HISTORY_DIR", history_dir):
                result = get_session_summary(session_id)

                assert result["first_message"] is None
