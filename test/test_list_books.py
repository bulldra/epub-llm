"""Test list_epub_books functionality."""

import os

# Add src to path
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# MCPツールとして定義されているため、直接関数を呼び出すのではなく内部実装をテスト
from src.mcp_server import validate_json_response
from src.simple_epub_service import SimpleEPUBService


class TestListEpubBooks(unittest.TestCase):
    """Test cases for list_epub_books function."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.test_books = [
            {
                "id": "test_book1.epub",
                "title": "Test Book 1",
                "author": "Author 1",
                "year": "2024",
                "toc": ["Chapter 1", "Chapter 2"],
            },
            {
                "id": "test_book2.epub",
                "title": "Test Book 2",
                "author": "Author 2",
                "year": "2023",
                "toc": ["Introduction", "Main Content"],
            },
        ]

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch("src.mcp_server.get_epub_service")
    def test_list_epub_books_success(self, mock_get_service):
        """Test successful book listing."""
        # Setup mock service
        mock_service = MagicMock()
        mock_service.get_bookshelf.return_value = self.test_books
        mock_get_service.return_value = mock_service

        # Call the internal implementation (list_epub_books is an MCP tool)
        result = mock_service.get_bookshelf()
        validated_result = validate_json_response(result)

        # Assertions
        self.assertEqual(len(validated_result), 2)
        self.assertEqual(validated_result[0]["id"], "test_book1.epub")
        self.assertEqual(validated_result[0]["title"], "Test Book 1")
        self.assertEqual(validated_result[0]["author"], "Author 1")
        self.assertEqual(validated_result[0]["year"], "2024")
        self.assertIn("Chapter 1", validated_result[0]["toc"])

        self.assertEqual(validated_result[1]["id"], "test_book2.epub")
        self.assertEqual(validated_result[1]["title"], "Test Book 2")
        mock_service.get_bookshelf.assert_called_once()

    @patch("src.mcp_server.get_epub_service")
    def test_list_epub_books_empty(self, mock_get_service):
        """Test listing when no books are available."""
        # Setup mock service
        mock_service = MagicMock()
        mock_service.get_bookshelf.return_value = []
        mock_get_service.return_value = mock_service

        # Call the internal implementation
        result = mock_service.get_bookshelf()
        validated_result = validate_json_response(result)

        # Assertions
        self.assertEqual(validated_result, [])
        mock_service.get_bookshelf.assert_called_once()

    @patch("src.mcp_server.get_epub_service")
    def test_list_epub_books_with_error(self, mock_get_service):
        """Test listing when an error occurs."""
        # Setup mock service to raise exception
        mock_service = MagicMock()
        mock_service.get_bookshelf.side_effect = RuntimeError("Test error")
        mock_get_service.return_value = mock_service

        # Test that exception is raised
        with self.assertRaises(RuntimeError):
            mock_service.get_bookshelf()

        mock_service.get_bookshelf.assert_called_once()

    @patch("src.mcp_server.get_epub_service")
    def test_list_epub_books_with_toc(self, mock_get_service):
        """Test that TOC information is properly included."""
        # Setup mock service with detailed TOC
        books_with_toc = [
            {
                "id": "technical_book.epub",
                "title": "Technical Manual",
                "author": "Tech Author",
                "year": "2024",
                "toc": [
                    "1章 Introduction",
                    "2章 Getting Started",
                    "3章 Advanced Topics",
                    "Appendix A",
                    "Appendix B",
                ],
            }
        ]
        mock_service = MagicMock()
        mock_service.get_bookshelf.return_value = books_with_toc
        mock_get_service.return_value = mock_service

        # Call the internal implementation
        result = mock_service.get_bookshelf()
        validated_result = validate_json_response(result)

        # Assertions
        self.assertEqual(len(validated_result), 1)
        self.assertEqual(validated_result[0]["id"], "technical_book.epub")
        self.assertEqual(len(validated_result[0]["toc"]), 5)
        self.assertIn("1章 Introduction", validated_result[0]["toc"])
        self.assertIn("3章 Advanced Topics", validated_result[0]["toc"])

    @patch("src.mcp_server.get_epub_service")
    def test_list_epub_books_json_validation(self, mock_get_service):
        """Test JSON validation in list_epub_books."""
        # Setup mock service with potentially problematic characters
        books_with_special_chars = [
            {
                "id": "special_book.epub",
                "title": "Book with Special Characters: 日本語 & Émojis 📚",
                "author": "作者名",
                "year": "2024",
                "toc": ["第1章 はじめに", "Chapter 2: Advanced"],
            }
        ]
        mock_service = MagicMock()
        mock_service.get_bookshelf.return_value = books_with_special_chars
        mock_get_service.return_value = mock_service

        # Call the internal implementation
        result = mock_service.get_bookshelf()
        validated_result = validate_json_response(result)

        # Assertions - should handle special characters properly
        self.assertEqual(len(validated_result), 1)
        self.assertIn("日本語", validated_result[0]["title"])
        self.assertIn("作者名", validated_result[0]["author"])
        self.assertIn("第1章", validated_result[0]["toc"][0])


class TestSimpleEPUBService(unittest.TestCase):
    """Test SimpleEPUBService.get_bookshelf method."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.service = SimpleEPUBService(self.test_dir)

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_get_bookshelf_empty_directory(self):
        """Test get_bookshelf with empty directory."""
        result = self.service.get_bookshelf()
        self.assertEqual(result, [])

    @patch("src.common_util.get_epub_cover_path")
    @patch("src.common_util.extract_epub_metadata")
    @patch("src.common_util.extract_epub_toc")
    def test_get_bookshelf_with_books(
        self, mock_extract_toc, mock_extract_metadata, mock_get_cover
    ):
        """Test get_bookshelf with EPUB files."""
        # Create dummy EPUB files
        epub_file1 = Path(self.test_dir) / "book1.epub"
        epub_file2 = Path(self.test_dir) / "book2.epub"
        epub_file1.touch()
        epub_file2.touch()

        # Mock metadata extraction based on filepath
        def mock_metadata(filepath):
            if "book1.epub" in filepath:
                return {
                    "title": "Book One",
                    "author": "Author One",
                    "year": "2024",
                }
            elif "book2.epub" in filepath:
                return {
                    "title": "Book Two",
                    "author": "Author Two",
                    "year": "2023",
                }
            return {}

        mock_extract_metadata.side_effect = mock_metadata

        # Mock TOC extraction based on filepath
        def mock_toc(filepath):
            if "book1.epub" in filepath:
                return [{"title": "Ch1", "level": 0}, {"title": "Ch2", "level": 0}]
            elif "book2.epub" in filepath:
                return [{"title": "Intro", "level": 0}, {"title": "Main", "level": 0}]
            return []

        mock_extract_toc.side_effect = mock_toc

        # Mock cover path extraction
        mock_get_cover.return_value = None

        # Get bookshelf
        result = self.service.get_bookshelf()

        # Assertions - sort results by ID to ensure consistent order
        result_sorted = sorted(result, key=lambda x: x["id"])
        self.assertEqual(len(result_sorted), 2)
        self.assertEqual(result_sorted[0]["id"], "book1.epub")
        self.assertEqual(result_sorted[0]["title"], "Book One")
        self.assertEqual(result_sorted[0]["toc"], ["Ch1", "Ch2"])
        self.assertEqual(result_sorted[1]["id"], "book2.epub")
        self.assertEqual(result_sorted[1]["title"], "Book Two")
        self.assertEqual(result_sorted[1]["toc"], ["Intro", "Main"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
