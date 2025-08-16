#!/usr/bin/env python3
"""
Test script for MLX-FAISS API endpoints.
"""

import json
import logging
import os
import sys
from pathlib import Path

import requests

# Add project root (parent of scripts) to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config_manager import AppConfig


def setup_logging():
    """Setup logging for test."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def test_mlx_api_endpoints():
    """Test MLX-FAISS API endpoints."""
    setup_logging()
    logger = logging.getLogger(__name__)

    # API base URL
    base_url = "http://localhost:8000"
    mlx_url = f"{base_url}/mlx"

    logger.info("=" * 60)
    logger.info("Testing MLX-FAISS API Endpoints")
    logger.info("=" * 60)

    # Test 1: Get stats
    logger.info("\n1. Testing GET /mlx/stats...")
    try:
        response = requests.get(f"{mlx_url}/stats")
        if response.status_code == 200:
            stats = response.json()
            logger.info("✓ Stats retrieved successfully")
            logger.info(f"   Response: {json.dumps(stats, indent=2)}")
        else:
            logger.error(f"✗ Failed to get stats: {response.status_code}")
    except Exception as e:
        logger.error(f"✗ Error getting stats: {e}")

    # Test 2: Rebuild index
    logger.info("\n2. Testing POST /mlx/index/rebuild...")
    try:
        response = requests.post(f"{mlx_url}/index/rebuild")
        if response.status_code == 200:
            result = response.json()
            logger.info("✓ Index rebuilt successfully")
            logger.info(f"   Response: {json.dumps(result, indent=2)}")
        else:
            logger.error(f"✗ Failed to rebuild index: {response.status_code}")
    except Exception as e:
        logger.error(f"✗ Error rebuilding index: {e}")

    # Test 3: Search
    logger.info("\n3. Testing POST /mlx/search...")
    test_queries = ["main character", "story", "important"]

    for query in test_queries:
        logger.info(f"\n   Testing search for: '{query}'")
        try:
            response = requests.post(
                f"{mlx_url}/search", json={"query": query, "top_k": 3}
            )
            if response.status_code == 200:
                results = response.json()
                logger.info(f"   ✓ Search completed: {len(results)} results")
                for i, result in enumerate(results[:2], 1):
                    if "error" not in result:
                        logger.info(
                            f"      Result {i}: Score={result.get('score', 0):.3f}"
                        )
            else:
                logger.error(f"   ✗ Search failed: {response.status_code}")
        except Exception as e:
            logger.error(f"   ✗ Error searching: {e}")

    # Test 4: Index specific book (if EPUB files exist)
    config = AppConfig()
    epub_dir = config.get("directories.epub_dir", "epubs")
    epub_files = [f for f in os.listdir(epub_dir) if f.endswith(".epub")]

    if epub_files:
        first_epub = epub_files[0]
        book_id = first_epub.replace(".epub", "")

        logger.info(f"\n4. Testing POST /mlx/index/book for {book_id}...")
        try:
            response = requests.post(
                f"{mlx_url}/index/book",
                json={"book_id": book_id, "epub_path": first_epub},
            )
            if response.status_code == 200:
                result = response.json()
                logger.info("✓ Book indexed successfully")
                logger.info(f"   Response: {result}")
            else:
                logger.error(f"✗ Failed to index book: {response.status_code}")
        except Exception as e:
            logger.error(f"✗ Error indexing book: {e}")

    # Test 5: Clear index
    logger.info("\n5. Testing DELETE /mlx/index...")
    try:
        response = requests.delete(f"{mlx_url}/index")
        if response.status_code == 200:
            result = response.json()
            logger.info("✓ Index cleared successfully")
            logger.info(f"   Response: {result}")
        else:
            logger.error(f"✗ Failed to clear index: {response.status_code}")
    except Exception as e:
        logger.error(f"✗ Error clearing index: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("MLX-FAISS API Test Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_mlx_api_endpoints()
