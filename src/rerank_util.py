"""
Re-ranking utilities for improving search result relevance.
"""

import logging
import re
from typing import Any


class QueryBasedReRanker:
    """Re-rank search results based on query-specific features."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def _calculate_query_text_overlap(self, query: str, text: str) -> float:
        """Calculate overlap between query and text."""
        query_words = set(re.findall(r"[ぁ-んァ-ヴー一-龯a-zA-Z0-9]+", query.lower()))
        text_words = set(re.findall(r"[ぁ-んァ-ヴー一-龯a-zA-Z0-9]+", text.lower()))

        if not query_words:
            return 0.0

        overlap = len(query_words.intersection(text_words))
        return overlap / len(query_words)

    def _calculate_text_quality_score(self, text: str) -> float:
        """Calculate text quality based on length and structure."""
        # Prefer texts that are not too short or too long
        length = len(text)

        if length < 50:
            length_score = 0.3
        elif length > 2000:
            length_score = 0.7
        else:
            length_score = 1.0

        # Check for structural elements (headers, etc.)
        structure_score = 0.0
        if re.search(r"^#+ ", text, re.MULTILINE):  # Headers
            structure_score += 0.2
        if "。" in text or "!" in text or "?" in text:  # Sentences
            structure_score += 0.3

        return min(length_score + structure_score, 1.0)

    def _calculate_context_diversity_penalty(
        self, results: list[dict[str, Any]]
    ) -> list[float]:
        """Calculate penalty for similar results to promote diversity."""
        penalties = [0.0] * len(results)

        for i, result_i in enumerate(results):
            text_i = result_i.get("text", "")
            words_i = set(re.findall(r"[ぁ-んァ-ヴー一-龯a-zA-Z0-9]+", text_i.lower()))

            for _j, result_j in enumerate(
                results[:i]
            ):  # Only compare with previous results
                text_j = result_j.get("text", "")
                words_j = set(
                    re.findall(r"[ぁ-んァ-ヴー一-龯a-zA-Z0-9]+", text_j.lower())
                )

                if words_i and words_j:
                    similarity = len(words_i.intersection(words_j)) / len(
                        words_i.union(words_j)
                    )
                    if similarity > 0.7:  # High similarity threshold
                        penalties[i] += 0.3 * similarity

        return penalties

    def rerank_results(
        self,
        query: str,
        results: list[dict[str, Any]],
        diversity_weight: float = 0.2,
        quality_weight: float = 0.1,
        overlap_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Re-rank results using multiple factors."""
        if not results:
            return results

        # Calculate additional features
        diversity_penalties = self._calculate_context_diversity_penalty(results)

        enhanced_results = []
        for i, result in enumerate(results):
            text = result.get("text", "")

            # Calculate additional scores
            overlap_score = self._calculate_query_text_overlap(query, text)
            quality_score = self._calculate_text_quality_score(text)
            diversity_penalty = diversity_penalties[i]

            # Combine scores
            original_score = result.get("score", result.get("combined_score", 0.0))

            rerank_score = (
                original_score
                + overlap_weight * overlap_score
                + quality_weight * quality_score
                - diversity_weight * diversity_penalty
            )

            enhanced_result = result.copy()
            enhanced_result.update(
                {
                    "rerank_score": rerank_score,
                    "overlap_score": overlap_score,
                    "quality_score": quality_score,
                    "diversity_penalty": diversity_penalty,
                    "original_score": original_score,
                }
            )

            enhanced_results.append(enhanced_result)

        # Sort by re-rank score
        enhanced_results.sort(key=lambda x: x["rerank_score"], reverse=True)

        self.logger.info(
            "[ReRank] Re-ranked %d results, score range: %.3f - %.3f",
            len(enhanced_results),
            enhanced_results[-1]["rerank_score"] if enhanced_results else 0,
            enhanced_results[0]["rerank_score"] if enhanced_results else 0,
        )

        return enhanced_results


class ContextCompressor:
    """Compress and filter context for better LLM performance."""

    def __init__(self, max_context_length: int = 8000):
        self.max_context_length = max_context_length
        self.logger = logging.getLogger(__name__)

    def _extract_key_sentences(
        self, text: str, query: str, max_sentences: int = 3
    ) -> str:
        """Extract key sentences from text that are most relevant to query."""
        sentences = re.split(r"[。！？]", text)
        query_words = set(re.findall(r"[ぁ-んァ-ヴー一-龯a-zA-Z0-9]+", query.lower()))

        sentence_scores = []
        for sentence in sentences:
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue

            sentence_words = set(
                re.findall(r"[ぁ-んァ-ヴー一-龯a-zA-Z0-9]+", sentence.lower())
            )

            if query_words and sentence_words:
                overlap = len(query_words.intersection(sentence_words))
                score = overlap / len(query_words)
                sentence_scores.append((score, sentence.strip()))

        # Sort by relevance and take top sentences
        sentence_scores.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s[1] for s in sentence_scores[:max_sentences] if s[1]]

        return "。".join(top_sentences) + "。" if top_sentences else text[:500]

    def compress_context(
        self, results: list[dict[str, Any]], query: str, preserve_book_info: bool = True
    ) -> str:
        """Compress context while preserving important information."""
        if not results:
            return ""

        compressed_parts = []
        current_length = 0

        # Group by book for better organization
        book_groups: dict[str, list[dict[str, Any]]] = {}
        for result in results:
            book_title = result.get("book_title", result.get("book_id", "Unknown"))
            if book_title not in book_groups:
                book_groups[book_title] = []
            book_groups[book_title].append(result)

        for book_title, book_results in book_groups.items():
            if current_length >= self.max_context_length:
                break

            book_parts = []
            for result in book_results[:3]:  # Limit results per book
                if current_length >= self.max_context_length:
                    break

                text = result.get("text", "")

                # Compress text if too long
                if len(text) > 400:
                    compressed_text = self._extract_key_sentences(text, query)
                else:
                    compressed_text = text

                # Add relevance indicator
                score = result.get("rerank_score", result.get("score", 0))
                if score > 0.7:
                    relevance = "高関連"
                elif score > 0.4:
                    relevance = "中関連"
                else:
                    relevance = "低関連"

                formatted_text = f"[{relevance}] {compressed_text}"
                book_parts.append(formatted_text)
                current_length += len(formatted_text)

            if book_parts and preserve_book_info:
                book_section = f"**{book_title}**\n" + "\n\n".join(book_parts)
                compressed_parts.append(book_section)
            elif book_parts:
                compressed_parts.extend(book_parts)

        final_result = "\n\n---\n\n".join(compressed_parts)

        self.logger.info(
            "[Compress] Compressed context from %d results to %d characters",
            len(results),
            len(final_result),
        )

        return final_result
