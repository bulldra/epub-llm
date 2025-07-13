"""
Enhanced EPUB service with smart RAG capabilities.
"""

import asyncio
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
from src.query_expansion_util import AdaptiveRAGStrategy, QueryExpander
from src.rerank_util import ContextCompressor, QueryBasedReRanker
from src.smart_rag_util import SmartRAGManager


class SearchComponents:
    """Container for search-related components."""

    def __init__(self, llm_manager: Any):
        self.query_expander = QueryExpander(llm_manager)
        self.reranker = QueryBasedReRanker()
        self.context_compressor = ContextCompressor(max_context_length=8000)
        self.adaptive_strategy = AdaptiveRAGStrategy()


class EnhancedEPUBService:
    """Enhanced EPUB service with smart RAG capabilities."""

    def __init__(
        self,
        epub_dir: str,
        cache_dir: str,
        smart_rag_manager: SmartRAGManager,
        llm_manager: Any,
    ):
        self.epub_dir = epub_dir
        self.cache_dir = cache_dir
        self.smart_rag_manager = smart_rag_manager
        self.logger = logging.getLogger(__name__)

        # Initialize search components
        self.search = SearchComponents(llm_manager)

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

    async def smart_search_books(
        self, book_ids: list[str], query: str, top_k: int = 10
    ) -> str:
        """Perform smart search across multiple books with query expansion."""
        if not book_ids or not query:
            return ""

        try:
            self.logger.info(
                "[Enhanced Search] Starting smart search for: %s", query[:50]
            )

            # Analyze and expand query
            strategy, search_queries = await self._prepare_search_strategy(query)

            # Execute searches and process results
            all_results = self._execute_hybrid_searches(
                search_queries, book_ids, strategy, query
            )

            # Process and format final context
            return self._process_and_format_results(all_results, query, strategy)

        except (OSError, ValueError, TypeError, RuntimeError) as e:
            self.logger.error("[Enhanced Search] Error in smart search: %s", e)
            return await self._fallback_search(book_ids, query, top_k)

    async def _prepare_search_strategy(
        self, query: str
    ) -> tuple[dict[str, Any], list[str]]:
        """Analyze query and prepare search strategy."""
        query_analysis = self.search.query_expander.analyze_query_intent(query)
        strategy = self.search.adaptive_strategy.determine_search_strategy(
            query_analysis
        )

        expansion_result = await self.search.query_expander.expand_query(query)
        search_queries = expansion_result["search_queries"]

        return strategy, search_queries

    def _execute_hybrid_searches(
        self,
        search_queries: list[str],
        book_ids: list[str],
        strategy: dict[str, Any],
        original_query: str,
    ) -> list[dict[str, Any]]:
        """Execute hybrid searches for all queries."""
        all_results = []
        for search_query in search_queries:
            results = self.smart_rag_manager.hybrid_search_with_book_weights(
                query=search_query,
                book_ids=book_ids,
                top_k=strategy["top_k"],
                semantic_weight=strategy["semantic_weight"],
                keyword_weight=strategy["keyword_weight"],
            )

            # Add query source information
            for result in results:
                result["source_query"] = search_query
                result["is_original"] = search_query == original_query

            all_results.extend(results)
        return all_results

    def _process_and_format_results(
        self, all_results: list[dict[str, Any]], query: str, strategy: dict[str, Any]
    ) -> str:
        """Process results and format final context."""
        # Remove duplicates and add metadata
        unique_results = self._deduplicate_results(all_results)
        enhanced_results = self._add_book_metadata(unique_results)

        # Re-rank if enabled
        if strategy["use_reranking"]:
            reranked_results = self.search.reranker.rerank_results(
                query=query,
                results=enhanced_results,
                diversity_weight=strategy["diversity_weight"],
            )
        else:
            reranked_results = enhanced_results

        # Generate final context
        final_results = reranked_results[: strategy["top_k"]]
        if strategy["context_compression"]:
            self.search.context_compressor.max_context_length = strategy[
                "max_context_length"
            ]
            context = self.search.context_compressor.compress_context(
                results=final_results, query=query
            )
        else:
            context = self._format_simple_context(final_results)

        self.logger.info(
            "[Enhanced Search] Generated context: %d chars from %d results",
            len(context),
            len(reranked_results),
        )
        return context

    def _deduplicate_results(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove duplicate results based on text content."""
        seen_texts = set()
        unique_results = []

        for result in results:
            text_key = (result.get("book_id", ""), result.get("index", 0))
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_results.append(result)

        return unique_results

    def _add_book_metadata(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add book metadata to results."""
        enhanced_results = []

        for result in results:
            book_id = result.get("book_id", "")
            epub_path = os.path.join(self.epub_dir, book_id)

            try:
                meta = extract_epub_metadata(epub_path)
                book_title = meta.get("title", book_id)
            except (OSError, ValueError, KeyError):
                book_title = book_id

            enhanced_result = result.copy()
            enhanced_result["book_title"] = book_title
            enhanced_results.append(enhanced_result)

        return enhanced_results

    def _format_simple_context(self, results: list[dict[str, Any]]) -> str:
        """Format context without compression."""
        if not results:
            return ""

        context_parts = []
        for result in results:
            book_title = result.get("book_title", "Unknown")
            text = result.get("text", "")
            score = result.get("rerank_score", result.get("combined_score", 0))

            context_parts.append(f"**{book_title}** (関連度: {score:.2f})\n{text}")

        return "\n\n---\n\n".join(context_parts)

    async def _fallback_search(
        self, book_ids: list[str], query: str, top_k: int
    ) -> str:
        """Fallback to basic search in case of errors."""
        try:
            self.smart_rag_manager.smart_search_with_metadata(query, book_ids, top_k)
            return self.smart_rag_manager.generate_smart_context(query, book_ids, top_k)
        except (OSError, ValueError, TypeError, RuntimeError) as e:
            self.logger.error("[Fallback Search] Error: %s", e)
            return ""

    def search_single_book(
        self, book_id: str, query: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Search within a single book using smart RAG."""
        # Use hybrid search for single book
        results = self.smart_rag_manager.hybrid_search(query, [book_id], top_k)

        # Add book metadata
        enhanced_results = self._add_book_metadata(results)

        # Re-rank for single book search
        reranked_results = self.search.reranker.rerank_results(query, enhanced_results)

        return reranked_results

    def get_context_for_query(
        self, query: str, book_ids: list[str], top_k: int = 10
    ) -> str:
        """Get context for a query using smart search (async wrapper)."""
        # Note: This is a sync wrapper for the async smart_search_books
        # In a real implementation, you might want to handle this differently
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.smart_search_books(book_ids, query, top_k)
            )
        except RuntimeError:
            # If no event loop is running, create a new one
            return asyncio.run(self.smart_search_books(book_ids, query, top_k))


class EnhancedChatService:
    """Enhanced chat service with smart RAG integration."""

    def __init__(
        self,
        epub_service: EnhancedEPUBService,
        llm_manager: Any,
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
        """Process a chat request with enhanced RAG."""
        try:
            # Add system message
            system_msg = {
                "role": "system",
                "content": "あなたは日本語のMarkdownで答えるアシスタントAIです。"
                "質問に対して、contextを活用して、正確かつ簡潔に答えてください。"
                "contextの内容を引用する場合は、引用符を使用し、出典を明記してください。",
            }
            messages.insert(0, system_msg)

            # Process RAG if books are selected
            context = ""
            if book_ids:
                self.logger.info("[Enhanced Chat] Processing %d books", len(book_ids))

                # Generate RAG query
                rag_query = self.llm_manager.generate_rag_query(messages)
                self.logger.info("[Enhanced Chat] Using RAG query: %s", rag_query)

                # Enhanced search for context
                context = await self.epub_service.smart_search_books(
                    book_ids, rag_query, top_k=7
                )

                # Also search with direct user query
                user_context = await self.epub_service.smart_search_books(
                    book_ids, messages[-1]["content"], top_k=3
                )

                # Merge contexts intelligently
                if user_context and context:
                    # Avoid duplication by checking overlap
                    if len(
                        set(context.split()) & set(user_context.split())
                    ) < 0.5 * len(context.split()):
                        context = f"{context}\n\n{user_context}"
                elif user_context:
                    context = user_context

                if context:
                    self.logger.info(
                        "[Enhanced Chat] Found context: %d characters", len(context)
                    )
                    messages[0]["content"] += f"\n\n## コンテキスト\n{context}\n"
                else:
                    self.logger.info("[Enhanced Chat] No context found")

            # Format prompt
            prompt = self.llm_manager.format_messages_as_prompt(messages)
            self.logger.info(
                "[Enhanced Chat] Final prompt ready (%d chars)", len(prompt)
            )

            return {
                "prompt": prompt,
                "context_size": len(messages[0]["content"]),
                "messages": messages,
            }

        except (OSError, ValueError, RuntimeError) as e:
            self.logger.error("Failed to process enhanced chat request: %s", e)
            raise
