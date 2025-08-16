"""
Simple EPUB service for MCP server with MLX-based embeddings + FAISS search.
"""

import logging
import os
from typing import Any

from src.common_util import get_book_list
from src.epub_util import extract_epub_metadata
from src.mlx_embedding_service import MLXEmbeddingService


class SimpleEPUBService:
    """Simple EPUB service for basic operations with MLX embedding search."""

    def __init__(self, epub_dir: str, embedding_model: str | None = None):
        from src.config_manager import AppConfig  # import-inside to respect guidelines

        self.epub_dir = epub_dir
        self.logger = logging.getLogger(__name__)
        # Initialize MLX embedding service (shared FAISS index under repo cache)
        cache_dir = os.path.join(os.path.dirname(self.epub_dir), "cache")
        # Fallback to config if not provided explicitly
        if not embedding_model:
            cfg = AppConfig()
            embedding_model = cfg.get("mlx.embedding_model")
            if not isinstance(embedding_model, str) or not embedding_model:
                raise RuntimeError(
                    "app_config.yaml の mlx.embedding_model が未設定です。必ずモデルIDを設定してください。"
                )
        self.embedding_service = MLXEmbeddingService(
            cache_dir, model_name=embedding_model
        )

    def ensure_index_loaded(self) -> None:
        """Ensure MLX-FAISS index is available; build if missing."""
        if self.embedding_service.load_index():
            return

        # Build index from all EPUBs if not present
        self.embedding_service.build_index(self.epub_dir)

    def ensure_book_indexed(self, book_id: str, epub_path: str) -> None:
        """Index a single book if not already in the index."""
        # Try to load or ensure index exists
        self.ensure_index_loaded()

        # Check stats to see if the book is already present
        stats = self.embedding_service.get_stats()
        books: dict[str, int] | None = (
            stats.get("books") if isinstance(stats, dict) else None
        )
        needs_index = not books or book_id not in books
        if needs_index:
            self.embedding_service.add_book(book_id, epub_path)
            self.embedding_service.save_index()

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

    def search_book_content(
        self, book_id: str, query: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Search for content in a specific book using MLX embeddings + FAISS."""
        epub_path = os.path.join(self.epub_dir, book_id)
        if not os.path.exists(epub_path):
            return [{"error": "Book not found"}]

        try:
            # Index the book if needed, then search limiting to the book
            book_id_no_ext = book_id.replace(".epub", "")
            self.ensure_book_indexed(book_id_no_ext, epub_path)
            results = self.embedding_service.search(
                query=query, top_k=top_k, book_id=book_id_no_ext
            )

            # Ensure book_id (filename) is present on results
            for result in results:
                if "error" not in result and "message" not in result:
                    result["book_id"] = book_id

            return results

        except (OSError, ValueError, RuntimeError) as e:
            self.logger.error("MLX search error for %s: %s", book_id, e)
            return [{"error": f"Search failed: {str(e)}"}]

    def search_all_books(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search for content across all books using MLX embeddings + FAISS.

        Args:
            query: 検索クエリ
            top_k: 返却する結果の最大件数（デフォルト: 10）

        Returns:
            検索結果のリスト。各結果は以下の情報を含む:
            - text: マッチしたテキスト内容
            - score: 関連度スコア
            - chunk_id: テキストチャンクのID
            - book_id: 書籍のファイル名
            - book_title: 書籍タイトル
        """
        try:
            # Ensure index is ready, then search across all books
            self.ensure_index_loaded()
            results = self.embedding_service.search(query=query, top_k=top_k)

            # Add book title information
            books = self.get_bookshelf()
            book_titles = {
                book["id"].replace(".epub", ""): book.get("title", "Unknown")
                for book in books
            }

            # Enhance results with book titles
            enhanced_results: list[dict[str, Any]] = []
            for result in results:
                if "error" not in result and "message" not in result:
                    book_key = result.get("book_id")
                    # Convert internal book_id to filename for UI consistency
                    result["book_title"] = book_titles.get(book_key, "Unknown")
                    result["book_id"] = f"{book_key}.epub" if book_key else None
                    enhanced_results.append(result)

            self.logger.debug("MLX全書籍検索完了: %d件の結果", len(enhanced_results))

            return enhanced_results if enhanced_results else results

        except (OSError, ValueError, RuntimeError) as e:
            self.logger.error("MLX全書籍検索エラー: %s", e)
            return [{"error": f"Search failed: {str(e)}"}]
