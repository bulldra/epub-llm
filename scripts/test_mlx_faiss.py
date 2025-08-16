#!/usr/bin/env python3
"""
Test script for MLX embedding service with FAISS integration.
"""

import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config_manager import AppConfig
from src.mlx_embedding_service import MLXEmbeddingService


def setup_logging():
    """Setup logging for test."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def test_mlx_embedding_service():
    """Test MLX embedding service functionality."""
    setup_logging()
    logger = logging.getLogger(__name__)

    # Load config
    config = AppConfig()
    cache_dir = config.get("directories.cache_dir", "cache")
    epub_dir = config.get("directories.epub_dir", "epubs")

    # Create directories if they don't exist
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(epub_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Testing MLX Embedding Service with FAISS")
    logger.info("=" * 60)

    # Initialize service
    logger.info("\n1. Initializing MLX Embedding Service...")
    service = MLXEmbeddingService(cache_dir)

    # Check for EPUB files
    epub_files = [f for f in os.listdir(epub_dir) if f.endswith(".epub")]
    if not epub_files:
        logger.warning(f"No EPUB files found in {epub_dir}")
        logger.info("Please add EPUB files to test the service.")
        return

    logger.info(f"Found {len(epub_files)} EPUB files")

    # Test loading existing index
    logger.info("\n2. Attempting to load existing index...")
    if service.load_index():
        logger.info("✓ Existing index loaded successfully")
        stats = service.get_stats()
        logger.info(f"   Index stats: {stats}")
    else:
        logger.info("No existing index found, will create new one")

    # Add first book to index
    if epub_files:
        first_epub = epub_files[0]
        book_id = first_epub.replace(".epub", "")
        epub_path = os.path.join(epub_dir, first_epub)

        logger.info(f"\n3. Adding book to index: {book_id}")
        try:
            service.add_book(book_id, epub_path)
            logger.info("✓ Book added successfully")

            # Save index
            logger.info("\n4. Saving index...")
            service.save_index()
            logger.info("✓ Index saved successfully")

        except Exception as e:
            logger.error(f"✗ Failed to add book: {e}")
            return

    # Test search functionality
    logger.info("\n5. Testing search functionality...")
    test_queries = [
        "main character",
        "beginning of the story",
        "important event",
        "love",
        "journey",
    ]

    for query in test_queries:
        logger.info(f"\n   Searching for: '{query}'")
        try:
            results = service.search(query, top_k=3)
            if results and not any("error" in r for r in results):
                logger.info(f"   ✓ Found {len(results)} results")
                for i, result in enumerate(results[:2], 1):
                    logger.info(f"      Result {i}: Score={result['score']:.3f}")
                    logger.info(f"      Text preview: {result['text'][:100]}...")
            else:
                logger.info("   No results found or error occurred")
        except Exception as e:
            logger.error(f"   ✗ Search failed: {e}")

    # Get final stats
    logger.info("\n6. Final index statistics:")
    stats = service.get_stats()
    for key, value in stats.items():
        if key != "books":
            logger.info(f"   {key}: {value}")
    if "books" in stats:
        logger.info(f"   Books indexed: {list(stats['books'].keys())}")

    logger.info("\n" + "=" * 60)
    logger.info("MLX-FAISS Integration Test Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_mlx_embedding_service()
