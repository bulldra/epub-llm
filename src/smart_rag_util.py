"""
Smart RAG utilities with hybrid search and advanced features.
"""

import json
import logging
import os
import re
from typing import Any

from rank_bm25 import BM25Okapi

from src.config_manager import get_config
from src.embedding_util import (
    ModelPair,
    build_faiss_index,
    create_embeddings_from_texts,
    load_embeddings,
    save_embeddings,
    search_similar,
)
from src.epub_util import extract_epub_metadata, extract_epub_text


class EmbeddingComponents:
    """Container for embedding-related components."""

    def __init__(self, embed_model: Any, embed_tokenizer: Any) -> None:
        self.model = embed_model
        self.tokenizer = embed_tokenizer
        self.model_pair = ModelPair(model=embed_model, tokenizer=embed_tokenizer)


class SearchIndexes:
    """Container for search indexes."""

    def __init__(self) -> None:
        self.bm25_indexes: dict[str, BM25Okapi] = {}
        self.tokenized_texts: dict[str, list[list[str]]] = {}


class SmartRAGManager:
    """Advanced RAG manager with hybrid search and intelligent ranking."""

    def __init__(
        self, embed_model: Any, embed_tokenizer: Any, cache_dir: str, epub_dir: str
    ):
        self.cache_dir = cache_dir
        self.epub_dir = epub_dir
        self.logger = logging.getLogger(__name__)
        self.config = get_config()

        # Initialize component containers
        self.embedding = EmbeddingComponents(embed_model, embed_tokenizer)
        self.indexes = SearchIndexes()

    def _tokenize_text(self, text: str) -> list[str]:
        """Tokenize text for BM25 search."""
        # 日本語対応の簡単なトークン化
        # ひらがな、カタカナ、漢字、英数字を考慮
        tokens = re.findall(r"[ぁ-んァ-ヴー一-龯a-zA-Z0-9]+", text.lower())
        return [token for token in tokens if len(token) > 1]

    def _build_bm25_index(self, book_id: str, texts: list[str]) -> None:
        """Build BM25 index for a book."""
        tokenized_docs = [self._tokenize_text(text) for text in texts]
        self.indexes.tokenized_texts[book_id] = tokenized_docs

        # Use config parameters for BM25
        bm25_params = self.config.get_bm25_params()
        self.indexes.bm25_indexes[book_id] = BM25Okapi(
            tokenized_docs,
            k1=bm25_params["k1"],
            b=bm25_params["b"],
            epsilon=bm25_params["epsilon"],
        )
        self.logger.info(
            "[Smart RAG] Built BM25 index for %s with %d docs", book_id, len(texts)
        )

    def _save_bm25_index(self, book_id: str) -> None:
        """Save BM25 index to cache."""
        if book_id not in self.indexes.bm25_indexes:
            return

        cache_path = os.path.join(self.cache_dir, f"{book_id}.bm25.json")

        # BM25のパラメータとトークン化テキストを保存
        bm25_data = {
            "tokenized_texts": self.indexes.tokenized_texts[book_id],
            "k1": self.indexes.bm25_indexes[book_id].k1,
            "b": self.indexes.bm25_indexes[book_id].b,
            "epsilon": self.indexes.bm25_indexes[book_id].epsilon,
        }

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(bm25_data, f, ensure_ascii=False)

    def _load_bm25_index(self, book_id: str) -> bool:
        """Load BM25 index from cache."""
        cache_path = os.path.join(self.cache_dir, f"{book_id}.bm25.json")

        if not os.path.exists(cache_path):
            return False

        try:
            with open(cache_path, encoding="utf-8") as f:
                bm25_data = json.load(f)

            self.indexes.tokenized_texts[book_id] = bm25_data["tokenized_texts"]
            self.indexes.bm25_indexes[book_id] = BM25Okapi(
                self.indexes.tokenized_texts[book_id],
                k1=bm25_data.get("k1", 1.2),
                b=bm25_data.get("b", 0.75),
                epsilon=bm25_data.get("epsilon", 0.25),
            )

            self.logger.info("[Smart RAG] Loaded BM25 index for %s", book_id)
            return True
        except (OSError, ValueError, KeyError) as e:
            self.logger.error(
                "[Smart RAG] Failed to load BM25 index for %s: %s", book_id, e
            )
            return False

    def ensure_embeddings_exist(self, book_id: str) -> bool:
        """Ensure embeddings and BM25 index exist for a book."""
        base_path = os.path.join(self.cache_dir, book_id)

        # Check if embeddings exist
        embeddings_exist = os.path.exists(base_path + ".npy") and os.path.exists(
            base_path + ".json"
        )

        # Check if BM25 index exists
        bm25_exists = self._load_bm25_index(book_id)

        if embeddings_exist and bm25_exists:
            self.logger.info("[Smart RAG] Using cached indexes for: %s", book_id)
            return True

        try:
            self.logger.info("[Smart RAG] Creating indexes for: %s", book_id)
            epub_path = os.path.join(self.epub_dir, book_id)

            # Extract text from EPUB
            text = extract_epub_text(epub_path, base_path + ".txt")
            if not text:
                self.logger.error("[Smart RAG] No text extracted from: %s", book_id)
                return False

            # Create text chunks
            chunks = self._create_smart_chunks(text)
            self.logger.info(
                "[Smart RAG] Created %d chunks for: %s", len(chunks), book_id
            )

            # Generate embeddings if needed
            if not embeddings_exist:
                embeddings = create_embeddings_from_texts(
                    chunks, self.embedding.model, self.embedding.tokenizer
                )
                save_embeddings(embeddings, chunks, base_path)
                self.logger.info("[Smart RAG] Saved embeddings for: %s", book_id)

            # Build BM25 index if needed
            if not bm25_exists:
                self._build_bm25_index(book_id, chunks)
                self._save_bm25_index(book_id)

            return True

        except (OSError, ValueError, RuntimeError) as e:
            self.logger.error(
                "[Smart RAG] Failed to create indexes for %s: %s", book_id, e
            )
            return False

    def _create_smart_chunks(self, text: str) -> list[str]:
        """Create intelligent text chunks considering sentence boundaries."""
        chunk_params = self.config.get_chunking_params()
        chunk_size = chunk_params["chunk_size"]
        overlap = chunk_params["overlap"]
        boundary_search = chunk_params["sentence_boundary_search"]

        chunks = []
        text_len = len(text)
        start = 0

        while start < text_len:
            end = min(start + chunk_size, text_len)

            # Try to end at sentence boundary
            if end < text_len:
                # Look for sentence endings within the configured range
                search_start = max(end - boundary_search, start)
                sentence_ends = []
                for pattern in ["。\n", "。", "!\n", "!", "?\n", "?"]:
                    pos = text.rfind(pattern, search_start, end)
                    if pos != -1:
                        sentence_ends.append(pos + len(pattern))

                if sentence_ends:
                    end = max(sentence_ends)

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)

            if end == text_len:
                break

            start += chunk_size - overlap

        return chunks

    def hybrid_search(
        self,
        query: str,
        book_ids: list[str],
        top_k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword search."""
        if not book_ids:
            return []

        all_results = []

        for book_id in book_ids:
            if not self.ensure_embeddings_exist(book_id):
                continue

            # Load embeddings
            base_path = os.path.join(self.cache_dir, book_id)
            embeddings, texts = load_embeddings(base_path)

            # Semantic search using embeddings
            index = build_faiss_index(embeddings)
            semantic_results = search_similar(
                query=query,
                model_pair=self.embedding.model_pair,
                index=index,
                texts=texts,
                top_k=top_k * 2,  # Get more candidates for re-ranking
            )

            # Keyword search using BM25
            if book_id not in self.indexes.bm25_indexes:
                self._load_bm25_index(book_id)

            keyword_scores = []
            if book_id in self.indexes.bm25_indexes:
                query_tokens = self._tokenize_text(query)
                bm25_scores = self.indexes.bm25_indexes[book_id].get_scores(
                    query_tokens
                )
                keyword_scores = list(enumerate(bm25_scores))
                keyword_scores.sort(key=lambda x: x[1], reverse=True)

            # Combine scores using weighted average
            combined_scores = {}

            # Add semantic scores
            for idx, score, text in semantic_results:
                if idx < len(texts):
                    combined_scores[idx] = {
                        "semantic_score": float(score),
                        "keyword_score": 0.0,
                        "text": text,
                        "book_id": book_id,
                    }

            # Add keyword scores
            for idx, score in keyword_scores[: top_k * 2]:
                if idx in combined_scores:
                    combined_scores[idx]["keyword_score"] = float(score)
                elif idx < len(texts):
                    combined_scores[idx] = {
                        "semantic_score": 0.0,
                        "keyword_score": float(score),
                        "text": texts[idx],
                        "book_id": book_id,
                    }

            # Calculate combined scores
            for idx, data in combined_scores.items():
                # Normalize scores (simple min-max normalization)
                semantic_score = data["semantic_score"]
                keyword_score_val = data["keyword_score"]
                semantic_norm = (
                    float(semantic_score)
                    if isinstance(semantic_score, int | float | str)
                    else 0.0
                )
                keyword_score = (
                    float(keyword_score_val)
                    if isinstance(keyword_score_val, int | float | str)
                    else 0.0
                )
                keyword_norm = (
                    min(keyword_score / 10.0, 1.0) if keyword_score > 0 else 0.0
                )

                combined_score = (
                    semantic_weight * semantic_norm + keyword_weight * keyword_norm
                )

                all_results.append(
                    {
                        "book_id": book_id,
                        "index": idx,
                        "combined_score": combined_score,
                        "semantic_score": data["semantic_score"],
                        "keyword_score": data["keyword_score"],
                        "text": data["text"],
                    }
                )

        # Sort by combined score and return top results
        all_results.sort(
            key=lambda x: (
                float(x["combined_score"])
                if isinstance(x["combined_score"], int | float | str)
                else 0.0
            ),
            reverse=True,
        )
        return all_results[:top_k]

    def smart_search_with_metadata(
        self, query: str, book_ids: list[str], top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Smart search with metadata and book title information."""
        results = self.hybrid_search(query, book_ids, top_k)

        # Add book metadata
        enhanced_results = []
        for result in results:
            book_id = result["book_id"]

            # Get book title
            epub_path = os.path.join(self.epub_dir, book_id)
            try:
                meta = extract_epub_metadata(epub_path)
                book_title = meta.get("title", book_id)
            except (OSError, ValueError, KeyError):
                book_title = book_id

            enhanced_results.append(
                {
                    "book_id": book_id,
                    "book_title": book_title,
                    "index": result["index"],
                    "score": result["combined_score"],
                    "semantic_score": result["semantic_score"],
                    "keyword_score": result["keyword_score"],
                    "text": result["text"],
                }
            )

        return enhanced_results

    def _calculate_book_relevance_weight(self, book_id: str, query: str) -> float:
        """Calculate book-specific relevance weight based on metadata."""
        try:
            epub_path = os.path.join(self.epub_dir, book_id)
            meta = extract_epub_metadata(epub_path)

            # Get weighting parameters from config
            weight_params = self.config.get_book_weighting_params()

            weight = 1.0
            query_lower = query.lower()

            # Boost weight if query matches title or author
            title = meta.get("title", "").lower()
            author = meta.get("author", "").lower()

            if title and any(word in title for word in query_lower.split()):
                weight += weight_params["title_match_bonus"]

            if author and any(word in author for word in query_lower.split()):
                weight += weight_params["author_match_bonus"]

            # Year-based relevance
            year = meta.get("year", "")
            if year and year.isdigit():
                year_int = int(year)
                if year_int > 2010:  # Recent books
                    weight += weight_params["recent_book_bonus"]
                elif year_int < 1990:  # Historical books
                    weight += weight_params["historical_book_bonus"]

            return min(weight, weight_params["max_weight"])

        except (OSError, ValueError, KeyError):
            return 1.0

    def hybrid_search_with_book_weights(
        self,
        query: str,
        book_ids: list[str],
        top_k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Enhanced hybrid search with book-specific weighting."""
        if not book_ids:
            return []

        all_results = []

        for book_id in book_ids:
            if not self.ensure_embeddings_exist(book_id):
                continue

            # Calculate book relevance weight
            book_weight = self._calculate_book_relevance_weight(book_id, query)

            # Load embeddings
            base_path = os.path.join(self.cache_dir, book_id)
            embeddings, texts = load_embeddings(base_path)

            # Semantic search
            index = build_faiss_index(embeddings)
            semantic_results = search_similar(
                query=query,
                model_pair=self.embedding.model_pair,
                index=index,
                texts=texts,
                top_k=top_k * 2,
            )

            # Keyword search
            if book_id not in self.indexes.bm25_indexes:
                self._load_bm25_index(book_id)

            keyword_scores = []
            if book_id in self.indexes.bm25_indexes:
                query_tokens = self._tokenize_text(query)
                bm25_scores = self.indexes.bm25_indexes[book_id].get_scores(
                    query_tokens
                )
                keyword_scores = list(enumerate(bm25_scores))
                keyword_scores.sort(key=lambda x: x[1], reverse=True)

            # Combine scores with book weighting
            combined_scores = {}

            for idx, score, text in semantic_results:
                if idx < len(texts):
                    combined_scores[idx] = {
                        "semantic_score": float(score) * book_weight,
                        "keyword_score": 0.0,
                        "text": text,
                        "book_id": book_id,
                        "book_weight": book_weight,
                    }

            for idx, score in keyword_scores[: top_k * 2]:
                if idx in combined_scores:
                    combined_scores[idx]["keyword_score"] = float(score) * book_weight
                elif idx < len(texts):
                    combined_scores[idx] = {
                        "semantic_score": 0.0,
                        "keyword_score": float(score) * book_weight,
                        "text": texts[idx],
                        "book_id": book_id,
                        "book_weight": book_weight,
                    }

            # Calculate weighted combined scores
            for idx, data in combined_scores.items():
                semantic_score = data["semantic_score"]
                keyword_score_val = data["keyword_score"]
                semantic_norm = (
                    float(semantic_score)
                    if isinstance(semantic_score, int | float | str)
                    else 0.0
                )
                keyword_score = (
                    float(keyword_score_val)
                    if isinstance(keyword_score_val, int | float | str)
                    else 0.0
                )
                keyword_norm = (
                    min(keyword_score / 10.0, 1.0) if keyword_score > 0 else 0.0
                )

                combined_score = (
                    semantic_weight * semantic_norm + keyword_weight * keyword_norm
                )

                all_results.append(
                    {
                        "book_id": book_id,
                        "index": idx,
                        "combined_score": combined_score,
                        "semantic_score": data["semantic_score"],
                        "keyword_score": data["keyword_score"],
                        "book_weight": data["book_weight"],
                        "text": data["text"],
                    }
                )

        all_results.sort(
            key=lambda x: (
                float(x["combined_score"])
                if isinstance(x["combined_score"], int | float | str)
                else 0.0
            ),
            reverse=True,
        )
        return all_results[:top_k]

    def generate_smart_context(
        self, query: str, book_ids: list[str], top_k: int = 10
    ) -> str:
        """Generate context with smart filtering and compression."""
        self.logger.info("[SmartRAG] Generating context for query: %s", query)
        results = self.smart_search_with_metadata(query, book_ids, top_k)

        if not results:
            return ""

        # Group results by book for better organization
        book_groups: dict[str, list[dict[str, Any]]] = {}
        book_weights: dict[str, float] = {}

        for result in results:
            book_title = result["book_title"]
            if book_title not in book_groups:
                book_groups[book_title] = []
                book_weights[book_title] = result.get("book_weight", 1.0)
            book_groups[book_title].append(result)

        # Sort books by their relevance weights
        sorted_books = sorted(
            book_groups.items(), key=lambda x: book_weights[x[0]], reverse=True
        )

        # Format context with book grouping and weights
        context_parts = []
        for book_title, book_results in sorted_books:
            book_parts = []
            weight_indicator = ""
            ui_params = self.config.get_ui_params()
            if book_weights[book_title] > 1.1 and ui_params["use_fire_emoji"]:
                weight_indicator = " 🔥"  # High relevance indicator

            for result in book_results:
                score_info = f"(類似度: {result['score']:.2f})"
                book_parts.append(f"{result['text']} {score_info}")

            if book_parts:
                context_parts.append(
                    f"**{book_title}**{weight_indicator}\n" + "\n\n".join(book_parts)
                )

        return "\n\n---\n\n".join(context_parts)
