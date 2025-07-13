"""EPUB processing utilities.

This module provides functions for extracting text and metadata from EPUB files,
including text extraction, cover image processing, and metadata parsing.
"""

import io
import logging
import os
from typing import TYPE_CHECKING, Any

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import ITEM_COVER, epub
from PIL import Image, UnidentifiedImageError

if TYPE_CHECKING:
    from ebooklib.epub import EpubBook
else:
    EpubBook = Any


def extract_epub_text(epub_path: str, cache_path: str) -> str:
    """Extract text content from an EPUB file with caching.

    Args:
        epub_path: Path to the EPUB file.
        cache_path: Path for caching the extracted text.

    Returns:
        The extracted text content as a string.
    """
    logging.info(
        "extract_epub_text: epub_path=%s, cache_path=%s", epub_path, cache_path
    )
    md_cache_path = os.path.splitext(cache_path)[0] + ".md"

    # Check for cached content
    if os.path.exists(md_cache_path):
        logging.info("extract_epub_text: cache hit %s", md_cache_path)
        with open(md_cache_path, encoding="utf-8") as f:
            return f.read()

    # Extract from EPUB
    logging.info("extract_epub_text: cache miss, extracting from %s", epub_path)
    book = epub.read_epub(epub_path)

    # Extract content and metadata
    md_body_lines = _extract_document_content(book)
    metadata = extract_epub_metadata(epub_path)

    # Build markdown content
    result = _build_markdown_content(metadata, md_body_lines)

    # Save to cache
    _save_to_cache(md_cache_path, result)
    return result


def _extract_document_content(book: "EpubBook") -> list[str]:
    """Extract text content from EPUB document items."""
    md_body_lines: list[str] = []

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")

            # Extract text from headers and paragraphs
            for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p"]):
                tag_name = getattr(tag, "name", None)
                if tag_name and tag_name.startswith("h"):
                    level = int(tag_name[1])
                    md_body_lines.append(f"{'#' * level} {tag.get_text(strip=True)}")
                elif tag_name == "p":
                    text = tag.get_text(strip=True)
                    if text:
                        md_body_lines.append(text)
            md_body_lines.append("")  # セクション区切り

    return md_body_lines


def _build_markdown_content(metadata: dict[str, str], content_lines: list[str]) -> str:
    """Build markdown content with YAML frontmatter."""
    md_lines = ["---"]

    for key, value in metadata.items():
        if value:
            md_lines.append(f"{key}: {value}")

    md_lines.extend(["---", ""])
    if metadata.get("title"):
        md_lines.append(f"# {metadata['title']}")
    md_lines.extend(content_lines)

    return "\n".join(md_lines)


def _save_to_cache(cache_path: str, content: str) -> None:
    """Save content to cache file."""
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(content)
    logging.info("extract_epub_text: saved cache to %s", cache_path)


def extract_epub_metadata(epub_path: str) -> dict[str, str]:
    """Extract metadata from an EPUB file.

    Args:
        epub_path: Path to the EPUB file.

    Returns:
        Dictionary containing metadata fields like title, creator, etc.
    """
    meta = {"title": "", "author": "", "year": ""}
    try:
        book = epub.read_epub(epub_path)
        title = book.get_metadata("DC", "title")
        if title and len(title) > 0:
            meta["title"] = title[0][0]
        author = book.get_metadata("DC", "creator")
        if author and len(author) > 0:
            meta["author"] = author[0][0]
        date = book.get_metadata("DC", "date")
        if date and len(date) > 0:
            meta["year"] = date[0][0]
    except (OSError, ValueError) as e:
        logging.warning("extract_epub_metadata: %s", e)
    return meta


def get_epub_cover_path(epub_path: str, cache_dir: str) -> str | None:
    """Get the cover image path for an EPUB file.

    Args:
        epub_path: Path to the EPUB file.
        cache_dir: Directory for caching cover images.

    Returns:
        Path to the cover image file, or None if not available.
    """
    cover_path = os.path.join(
        cache_dir,
        os.path.basename(epub_path) + ".cover.jpg",
    )
    if os.path.exists(cover_path):
        return cover_path
    try:
        return extract_and_save_cover(epub_path, cover_path)
    except (
        epub.EpubException,
        OSError,
        ValueError,
        UnidentifiedImageError,
    ) as e:
        logging.error("cover extract error: %s: %s", epub_path, e)
        return None


def extract_and_save_cover(epub_path: str, cover_path: str) -> str | None:
    """Extract and save the cover image from an EPUB file.

    Args:
        epub_path: Path to the EPUB file.
        cover_path: Path where the cover image should be saved.

    Returns:
        Path to the saved cover image, or None if extraction failed.
    """
    book = epub.read_epub(epub_path)
    cover_id = None
    cover_meta = book.get_metadata("OPF", "cover")
    if cover_meta and len(cover_meta) > 0:
        meta_dict = cover_meta[0][1]
        if isinstance(meta_dict, dict) and "content" in meta_dict:
            cover_id = meta_dict["content"]
    cover_item = None
    if cover_id is not None:
        cover_item = book.get_item_with_id(cover_id)
    if cover_item is None:
        for item in book.get_items():
            if item.get_type() == ITEM_COVER:
                cover_item = item
                break
    if cover_item is not None:
        try:
            cover_bytes = cover_item.get_content()
            img = Image.open(io.BytesIO(cover_bytes)).convert("RGB")
            os.makedirs(os.path.dirname(cover_path), exist_ok=True)
            img.save(cover_path, "JPEG")
            return cover_path
        except (OSError, ValueError, UnidentifiedImageError) as err:
            logging.error("cover save error: %s: %s", cover_path, err)
            return None
    return None
