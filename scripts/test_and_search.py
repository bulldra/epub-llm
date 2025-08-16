#!/usr/bin/env python3
"""
Test script for AND search functionality.
"""

import logging
import os
import sys
from pathlib import Path

# Add project root (parent of scripts) to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config_manager import AppConfig
from src.mlx_embedding_service import MLXEmbeddingService


def setup_logging():
    """Setup logging for test."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def test_and_search():
    """Test AND search functionality."""
    setup_logging()
    logger = logging.getLogger(__name__)

    # Load config
    config = AppConfig()
    cache_dir = config.get("directories.cache_dir", "cache")
    epub_dir = config.get("directories.epub_dir", "epubs")

    logger.info("=" * 60)
    logger.info("Testing AND Search Functionality")
    logger.info("=" * 60)

    # Test queries for AND search
    test_queries = [
        # Single term queries (normal search)
        ("Python", "Single term search"),
        ("programming", "Single term search"),
        # Multi-term queries (AND search)
        ("Python programming", "AND search: Python AND programming"),
        ("machine learning", "AND search: machine AND learning"),
        ("data analysis", "AND search: data AND analysis"),
        ("web development", "AND search: web AND development"),
        ("software engineering", "AND search: software AND engineering"),
    ]

    # Test 1: MLX Embedding Service
    logger.info("\n1. Testing MLX Embedding Service AND Search")
    logger.info("-" * 40)

    mlx_service = MLXEmbeddingService(cache_dir)

    # Try to load index
    if not mlx_service.load_index():
        logger.warning("No MLX index found. Building index from EPUB files...")

        # Build index from EPUB files
        epub_files = [f for f in os.listdir(epub_dir) if f.endswith(".epub")]
        if epub_files:
            for epub_file in epub_files[:3]:  # Index first 3 books for testing
                book_id = epub_file.replace(".epub", "")
                epub_path = os.path.join(epub_dir, epub_file)
                try:
                    mlx_service.add_book(book_id, epub_path)
                    logger.info(f"Indexed: {book_id}")
                except Exception as e:
                    logger.error(f"Failed to index {book_id}: {e}")

            mlx_service.save_index()

    # Test searches
    for query, description in test_queries:
        logger.info(f"\nQuery: '{query}' ({description})")
        try:
            results = mlx_service.search(query, top_k=3)
            if results and not any("error" in r for r in results):
                logger.info(f"   ✓ Found {len(results)} results")
                if results:
                    # Show first result preview
                    first_result = results[0]
                    text_preview = first_result["text"][:100].replace("\n", " ")
                    logger.info(
                        f"   Top result (score: {first_result['score']:.3f}): {text_preview}..."
                    )
            else:
                logger.info("   No results found")
        except Exception as e:
            logger.error(f"   ✗ Search failed: {e}")

    # Test 2: FAISS TF-IDF Service
    # Test 2: Specific AND search examples
    logger.info("\n\n2. Testing Specific AND Search Cases")
    logger.info("-" * 40)

    # Test with 3+ terms
    complex_queries = [
        "Python web framework",
        "machine learning algorithm",
        "data science visualization",
    ]

    for query in complex_queries:
        terms = query.split()
        logger.info(f"\nComplex AND query: '{query}' (must contain all: {terms})")

        try:
            results = mlx_service.search(query, top_k=5)
            if results and not any("error" in r for r in results):
                logger.info(f"   Found {len(results)} results")

                # Verify all terms are present in results
                for i, result in enumerate(results[:2], 1):
                    text_lower = result["text"].lower()
                    all_present = all(term.lower() in text_lower for term in terms)
                    status = "✓" if all_present else "✗"
                    logger.info(f"   Result {i}: {status} All terms present")
                    if not all_present:
                        missing = [t for t in terms if t.lower() not in text_lower]
                        logger.warning(f"      Missing terms: {missing}")
            else:
                logger.info("   No results found")
        except Exception as e:
            logger.error(f"   ✗ Search failed: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("AND Search Test Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_and_search()
