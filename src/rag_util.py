"""
RAG (Retrieval-Augmented Generation) utility functions.
"""

import logging
import os
from typing import Any

import numpy as np

from src.common_util import create_text_chunks
from src.embedding_util import (
    ModelPair,
    build_faiss_index,
    create_embeddings_from_texts,
    load_embeddings,
    save_embeddings,
    search_similar,
)
from src.epub_util import extract_epub_metadata, extract_epub_text


class RAGManager:
    """Manages RAG operations including embedding creation and similarity search."""

    def __init__(
        self, embed_model: Any, embed_tokenizer: Any, cache_dir: str, epub_dir: str
    ):
        self.embed_model = embed_model
        self.embed_tokenizer = embed_tokenizer
        self.cache_dir = cache_dir
        self.epub_dir = epub_dir
        self.logger = logging.getLogger(__name__)
        self.model_pair = ModelPair(model=embed_model, tokenizer=embed_tokenizer)

    def ensure_embeddings_exist(self, book_id: str) -> bool:
        """Ensure embeddings exist for a book, creating them if necessary."""
        base_path = os.path.join(self.cache_dir, book_id)

        if os.path.exists(base_path + ".npy") and os.path.exists(base_path + ".json"):
            self.logger.info("[RAG] Using cached embeddings for: %s", book_id)
            return True

        try:
            self.logger.info("[RAG] Creating embeddings for: %s", book_id)
            epub_path = os.path.join(self.epub_dir, book_id)

            # Extract text from EPUB
            text = extract_epub_text(epub_path, base_path + ".txt")
            if not text:
                self.logger.error("[RAG] No text extracted from: %s", book_id)
                return False

            # Create text chunks
            chunks = self._create_text_chunks(text)
            self.logger.info("[RAG] Created %d chunks for: %s", len(chunks), book_id)

            # Generate embeddings
            embeddings = create_embeddings_from_texts(
                chunks, self.embed_model, self.embed_tokenizer
            )

            # Save embeddings
            save_embeddings(embeddings, chunks, base_path)
            self.logger.info("[RAG] Saved embeddings for: %s", book_id)
            return True

        except (OSError, ValueError, RuntimeError) as e:
            self.logger.error(
                "[RAG] Failed to create embeddings for %s: %s", book_id, e
            )
            return False

    def _create_text_chunks(
        self, text: str, chunk_size: int = 4000, overlap: int = 500
    ) -> list[str]:
        """Create overlapping text chunks from text."""
        return create_text_chunks(text, chunk_size, overlap)

    def load_book_embeddings(self, book_id: str) -> tuple[np.ndarray, list[str]] | None:
        """Load embeddings for a single book."""
        if not self.ensure_embeddings_exist(book_id):
            return None

        try:
            base_path = os.path.join(self.cache_dir, book_id)
            embeddings, texts = load_embeddings(base_path)
            self.logger.info(
                "[RAG] Loaded embeddings for %s: %s, %d texts",
                book_id,
                embeddings.shape,
                len(texts),
            )
            return embeddings, texts
        except (OSError, ValueError, RuntimeError) as e:
            self.logger.error("[RAG] Failed to load embeddings for %s: %s", book_id, e)
            return None

    def load_multiple_books_embeddings(
        self, book_ids: list[str]
    ) -> tuple[np.ndarray | None, list[str], list[str], dict[str, str]]:
        """Load and combine embeddings from multiple books."""
        all_embeddings = []
        all_texts = []
        all_book_ids = []
        book_id_to_title = {}

        for book_id in book_ids:
            result = self.load_book_embeddings(book_id)
            if result is not None:
                embeddings, texts = result
                all_embeddings.append(embeddings)
                all_texts.extend(texts)
                all_book_ids.extend([book_id] * len(texts))

                # Get book title
                epub_path = os.path.join(self.epub_dir, book_id)
                try:
                    meta = extract_epub_metadata(epub_path)
                    title = meta.get("title", book_id)
                    book_id_to_title[book_id] = title
                except (OSError, ValueError, KeyError):
                    book_id_to_title[book_id] = book_id

        if not all_embeddings:
            return None, [], [], {}

        # Combine all embeddings
        combined_embeddings = np.concatenate(all_embeddings, axis=0)

        return combined_embeddings, all_texts, all_book_ids, book_id_to_title

    def search_context(
        self, query: str, book_ids: list[str], top_k: int = 10
    ) -> tuple[str, list[tuple[int, float, str]]]:
        """Search for relevant context across multiple books."""
        if not book_ids:
            return "", []

        # Load embeddings for all books
        embeddings, texts, _, _ = self.load_multiple_books_embeddings(book_ids)

        if embeddings is None:
            self.logger.warning("[RAG] No embeddings loaded for books: %s", book_ids)
            return "", []

        # Build search index
        index = build_faiss_index(embeddings)

        # Search for similar content
        results = search_similar(
            query=query,
            model_pair=self.model_pair,
            index=index,
            texts=texts,
            top_k=top_k,
        )

        # Format context
        context = "\\n---\\n".join([r[2] for r in results])

        self.logger.info(
            "[RAG] Found %d context chunks for query: %s", len(results), query[:50]
        )

        return context, results

    def search_context_with_metadata(
        self, query: str, book_ids: list[str], top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Search for relevant context and return structured metadata."""
        if not book_ids:
            return []

        # Load embeddings for all books
        embeddings, texts, book_id_texts, book_id_to_title = (
            self.load_multiple_books_embeddings(book_ids)
        )

        if embeddings is None:
            self.logger.warning("[RAG] No embeddings loaded for books: %s", book_ids)
            return []

        # Build search index
        index = build_faiss_index(embeddings)

        # Search for similar content
        results = search_similar(
            query=query,
            model_pair=self.model_pair,
            index=index,
            texts=texts,
            top_k=top_k,
        )

        # Create structured results
        context_items = []
        for idx, score, text in results:
            book_id = book_id_texts[idx]
            book_title = book_id_to_title.get(book_id, book_id)

            context_items.append(
                {
                    "book_id": book_id,
                    "book_title": book_title,
                    "index": idx,
                    "score": float(score),
                    "text": text,
                }
            )

        return context_items

    def search_single_book(
        self, book_id: str, query: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Search within a single book."""
        result = self.load_book_embeddings(book_id)
        if result is None:
            return [{"error": "Embeddings not found. Process the book first."}]

        embeddings, texts = result

        try:
            index = build_faiss_index(embeddings)
            results = search_similar(
                query=query,
                model_pair=self.model_pair,
                index=index,
                texts=texts,
                top_k=top_k,
            )

            return [
                {"index": r[0], "score": float(r[1]), "text": r[2], "book_id": book_id}
                for r in results
            ]
        except (OSError, ValueError, RuntimeError) as e:
            self.logger.error("[RAG] Search error for book %s: %s", book_id, e)
            return [{"error": str(e)}]
