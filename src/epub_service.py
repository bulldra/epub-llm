"""
EPUB service layer for business logic separation.
"""

import logging
import os
from typing import Any

from src.common_util import delete_book_files, get_book_list
from src.epub_util import extract_epub_metadata
from src.history_util import (
    delete_history,
    get_all_sessions,
    get_session_summary,
    load_history,
    load_session_data,
    save_history,
)
from src.llm_util import LLMManager
from src.rag_util import RAGManager


class EPUBService:
    """Service layer for EPUB-related operations."""

    def __init__(
        self,
        epub_dir: str,
        cache_dir: str,
        rag_manager: RAGManager,
    ):
        self.epub_dir = epub_dir
        self.cache_dir = cache_dir
        self.rag_manager = rag_manager
        self.logger = logging.getLogger(__name__)

    def get_bookshelf(self) -> list[dict[str, Any]]:
        """Get list of available EPUB books."""
        return get_book_list(self.epub_dir)

    def get_book_metadata(self, book_id: str) -> dict[str, Any]:
        """Get metadata for a specific book."""
        epub_path = os.path.join(self.epub_dir, book_id)
        if not os.path.exists(epub_path):
            return {"error": "Book not found"}

        try:
            return extract_epub_metadata(epub_path)
        except (OSError, ValueError, KeyError) as e:
            self.logger.error("Failed to extract metadata for %s: %s", book_id, e)
            return {"error": str(e)}

    def delete_book(self, book_id: str) -> dict[str, Any]:
        """Delete a book and its associated files."""
        try:
            delete_book_files(self.epub_dir, self.cache_dir, book_id)
            return {"result": "ok"}
        except (OSError, ValueError) as e:
            self.logger.error("Failed to delete book %s: %s", book_id, e)
            return {"error": str(e)}

    def search_books(self, book_ids: list[str], query: str, top_k: int = 10) -> str:
        """Search for content across multiple books and return formatted context."""
        context_items = self.rag_manager.search_context_with_metadata(
            query, book_ids, top_k
        )

        # Format context as a single string with book metadata
        context_parts = []
        for item in context_items:
            book_title = item.get("book_title", item.get("book_id", "Unknown"))
            text = item.get("text", "")
            position = item.get("index", 0)

            context_parts.append(f"**{book_title}** (位置: {position})\n{text}")

        return "\n\n---\n\n".join(context_parts)

    def search_single_book(
        self, book_id: str, query: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Search within a single book."""
        return self.rag_manager.search_single_book(book_id, query, top_k)

    def get_context_for_query(
        self, query: str, book_ids: list[str], top_k: int = 10
    ) -> str:
        """Get context for a query using MCP-style interface."""
        return self.search_books(book_ids, query, top_k)


class ChatService:
    """Service layer for chat-related operations."""

    def __init__(
        self,
        epub_service: EPUBService,
        llm_manager: LLMManager,
    ):
        self.epub_service = epub_service
        self.llm_manager = llm_manager
        self.logger = logging.getLogger(__name__)

    def get_all_sessions(self) -> list[dict[str, Any]]:
        """Get all chat sessions with summaries."""
        sessions = get_all_sessions()
        summaries = []

        for session_id in sessions:
            summary = get_session_summary(session_id)
            if summary:
                summaries.append(summary)

        return summaries

    def get_session_history(self, session_id: str) -> list[dict[str, Any]] | None:
        """Get chat history for a session."""
        return load_history(session_id)

    def get_session_data(self, session_id: str) -> dict[str, Any] | None:
        """Get full session data including messages and book selection."""
        return load_session_data(session_id)

    def save_session(
        self, session_id: str, messages: list[dict[str, Any]], book_ids: list[str]
    ) -> dict[str, Any]:
        """Save session data."""
        try:
            save_history(session_id, messages, book_ids)
            return {"result": "ok"}
        except (OSError, ValueError, RuntimeError) as e:
            self.logger.error("Failed to save session %s: %s", session_id, e)
            return {"error": str(e)}

    def delete_session(self, session_id: str) -> dict[str, Any]:
        """Delete a chat session."""
        try:
            success = delete_history(session_id)
            if success:
                return {"result": "ok"}
            return {"error": "Session not found"}
        except (OSError, ValueError) as e:
            self.logger.error("Failed to delete session %s: %s", session_id, e)
            return {"error": str(e)}

    async def process_chat_request(
        self, messages: list[dict[str, str]], book_ids: list[str]
    ) -> dict[str, Any]:
        """Process a chat request and return context info."""
        try:
            # Add system message
            system_msg = {
                "role": "system",
                "content": "あなたは日本語のMarkdownで答えるアシスタントAIです。",
            }
            messages.insert(0, system_msg)

            # Process RAG if books are selected
            context = ""
            if book_ids:
                self.logger.info("[RAG] Processing %d books", len(book_ids))

                # Generate RAG query
                rag_query = self.llm_manager.generate_rag_query(messages)
                self.logger.info("[Chat] Using RAG query: %s", rag_query)

                # Search for context
                context = self.epub_service.search_books(book_ids, rag_query, top_k=7)

                # Also search with direct user query
                user_context = self.epub_service.search_books(
                    book_ids, messages[-1]["content"], top_k=3
                )

                # Merge contexts
                if user_context:
                    context = (
                        f"{context}\n\n{user_context}" if context else user_context
                    )

                if context:
                    self.logger.info("[RAG] Found context: %d characters", len(context))
                    messages[0]["content"] += f"\n\n## コンテキスト\n{context}\n"
                else:
                    self.logger.info("[RAG] No context found")

            # Format prompt
            prompt = self.llm_manager.format_messages_as_prompt(messages)
            self.logger.info("[Chat] Final prompt ready (%d chars)", len(prompt))

            return {
                "prompt": prompt,
                "context_size": len(messages[0]["content"]),
                "messages": messages,
            }

        except (OSError, ValueError, RuntimeError) as e:
            self.logger.error("Failed to process chat request: %s", e)
            raise
