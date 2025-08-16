"""Tests for epub_util module."""

import os
import tempfile
from unittest.mock import MagicMock, patch

from src.epub_util import (
    extract_and_save_cover,
    extract_epub_metadata,
    extract_epub_text,
    get_epub_cover_path,
    stream_epub_markdown,
)


class TestEpubUtil:
    """Test cases for epub_util functions."""

    def test_extract_epub_text_with_cache_hit(self):
        """Test that cached text is returned when cache file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            epub_path = os.path.join(temp_dir, "test.epub")
            cache_path = os.path.join(temp_dir, "test_cache")
            md_cache_path = os.path.join(temp_dir, "test_cache.md")

            # Create a mock cache file
            cached_content = "# Test Content\nThis is cached content."
            with open(md_cache_path, "w", encoding="utf-8") as f:
                f.write(cached_content)

            result = extract_epub_text(epub_path, cache_path)
            assert result == cached_content

    def test_extract_epub_text_without_cache(self):
        """Test text extraction when no cache exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            epub_path = os.path.join(temp_dir, "test.epub")
            cache_path = os.path.join(temp_dir, "test_cache")

            # Mock the epub reading
            mock_book = MagicMock()
            mock_item = MagicMock()
            mock_item.get_type.return_value = 1  # ebooklib.ITEM_DOCUMENT
            mock_item.get_content.return_value = (
                b"<html><body><h1>Title</h1><p>Content</p></body></html>"
            )
            mock_book.get_items.return_value = [mock_item]

            with (
                patch("src.epub_util.epub.read_epub", return_value=mock_book),
                patch(
                    "src.epub_util.extract_epub_metadata",
                    return_value={"title": "Test", "author": "Author", "year": "2024"},
                ),
                patch("src.epub_util.ebooklib.ITEM_DOCUMENT", 1),
            ):
                result = extract_epub_text(epub_path, cache_path)

                assert "# Test" in result
                assert "# Title" in result
                assert "Content" in result
                assert os.path.exists(cache_path + ".md")

    def test_extract_epub_text_with_image(self):
        """Images in EPUB should be embedded as data URIs in markdown."""
        with tempfile.TemporaryDirectory() as temp_dir:
            epub_path = os.path.join(temp_dir, "img.epub")
            cache_path = os.path.join(temp_dir, "img_cache")

            mock_book = MagicMock()
            mock_doc = MagicMock()
            mock_doc.get_type.return_value = 1
            mock_doc.get_content.return_value = (
                b"<html><body><img src='a.png' alt='pic'/></body></html>"
            )
            mock_book.get_items.return_value = [mock_doc]

            mock_img = MagicMock()
            mock_img.get_content.return_value = b"binary"
            mock_img.media_type = "image/png"
            mock_book.get_item_with_href.return_value = mock_img

            with (
                patch("src.epub_util.epub.read_epub", return_value=mock_book),
                patch("src.epub_util.extract_epub_metadata", return_value={}),
                patch("src.epub_util.ebooklib.ITEM_DOCUMENT", 1),
            ):
                result = extract_epub_text(epub_path, cache_path)
                assert "![pic](data:image/png;base64," in result

    def test_stream_epub_markdown(self):
        """Streaming returns sequential chunk IDs with text."""
        with tempfile.TemporaryDirectory() as temp_dir:
            epub_path = os.path.join(temp_dir, "stream.epub")
            cache_path = os.path.join(temp_dir, "stream_cache")

            sample_md = "\n\n".join(["para" + str(i) for i in range(20)])

            with patch("src.epub_util.extract_epub_text", return_value=sample_md):
                chunks = list(stream_epub_markdown(epub_path, cache_path, max_chars=50))
            assert len(chunks) > 1
            assert chunks[0]["chunk_id"] == 0
            assert all("text" in c for c in chunks)

    def test_extract_epub_metadata_success(self):
        """Test successful metadata extraction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            epub_path = os.path.join(temp_dir, "test.epub")

            mock_book = MagicMock()
            mock_book.get_metadata.side_effect = [
                [("Test Title", {})],  # title
                [("Test Author", {})],  # creator
                [("2024", {})],  # date
            ]

            with patch("src.epub_util.epub.read_epub", return_value=mock_book):
                result = extract_epub_metadata(epub_path)

                assert result["title"] == "Test Title"
                assert result["author"] == "Test Author"
                assert result["year"] == "2024"

    def test_extract_epub_metadata_with_error(self):
        """Test metadata extraction with file errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            epub_path = os.path.join(temp_dir, "nonexistent.epub")

            with patch(
                "src.epub_util.epub.read_epub", side_effect=OSError("File not found")
            ):
                result = extract_epub_metadata(epub_path)

                assert result == {"title": "", "author": "", "year": ""}

    def test_get_epub_cover_path_with_existing_cover(self):
        """Test cover path retrieval when cover already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            epub_path = os.path.join(temp_dir, "test.epub")
            cache_dir = temp_dir
            cover_path = os.path.join(cache_dir, "test.epub.cover.jpg")

            # Create a mock cover file
            with open(cover_path, "w", encoding="utf-8") as f:
                f.write("mock cover")

            result = get_epub_cover_path(epub_path, cache_dir)
            assert result == cover_path

    def test_get_epub_cover_path_extraction_needed(self):
        """Test cover path when extraction is needed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            epub_path = os.path.join(temp_dir, "test.epub")
            cache_dir = temp_dir
            expected_cover_path = os.path.join(cache_dir, "test.epub.cover.jpg")

            with patch(
                "src.epub_util.extract_and_save_cover", return_value=expected_cover_path
            ):
                result = get_epub_cover_path(epub_path, cache_dir)
                assert result == expected_cover_path

    def test_get_epub_cover_path_extraction_fails(self):
        """Test cover path when extraction fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            epub_path = os.path.join(temp_dir, "test.epub")
            cache_dir = temp_dir

            with patch(
                "src.epub_util.extract_and_save_cover",
                side_effect=OSError("Extraction failed"),
            ):
                result = get_epub_cover_path(epub_path, cache_dir)
                assert result is None

    def test_extract_and_save_cover_success(self):
        """Test successful cover extraction and saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            epub_path = os.path.join(temp_dir, "test.epub")
            cover_path = os.path.join(temp_dir, "cover.jpg")

            # Mock the epub and cover data
            mock_book = MagicMock()
            mock_book.get_metadata.return_value = [("", {"content": "cover-id"})]

            mock_cover_item = MagicMock()
            mock_cover_item.get_content.return_value = b"fake image data"
            mock_book.get_item_with_id.return_value = mock_cover_item

            # Mock PIL Image
            mock_image = MagicMock()
            mock_image.convert.return_value = mock_image

            with (
                patch("src.epub_util.epub.read_epub", return_value=mock_book),
                patch("src.epub_util.Image.open", return_value=mock_image),
                patch("os.makedirs"),
            ):
                result = extract_and_save_cover(epub_path, cover_path)

                assert result == cover_path
                mock_image.save.assert_called_once_with(cover_path, "JPEG")

    def test_extract_and_save_cover_no_cover_found(self):
        """Test cover extraction when no cover is found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            epub_path = os.path.join(temp_dir, "test.epub")
            cover_path = os.path.join(temp_dir, "cover.jpg")

            mock_book = MagicMock()
            mock_book.get_metadata.return_value = []
            mock_book.get_item_with_id.return_value = None
            mock_book.get_items.return_value = []

            with patch("src.epub_util.epub.read_epub", return_value=mock_book):
                result = extract_and_save_cover(epub_path, cover_path)
                assert result is None

    def test_extract_and_save_cover_image_error(self):
        """Test cover extraction with image processing error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            epub_path = os.path.join(temp_dir, "test.epub")
            cover_path = os.path.join(temp_dir, "cover.jpg")

            mock_book = MagicMock()
            mock_book.get_metadata.return_value = [("", {"content": "cover-id"})]

            mock_cover_item = MagicMock()
            mock_cover_item.get_content.return_value = b"fake image data"
            mock_book.get_item_with_id.return_value = mock_cover_item

            with (
                patch("src.epub_util.epub.read_epub", return_value=mock_book),
                patch("src.epub_util.Image.open", side_effect=OSError("Image error")),
            ):
                result = extract_and_save_cover(epub_path, cover_path)
                assert result is None
