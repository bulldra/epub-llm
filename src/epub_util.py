import io
import logging
import os
from typing import List, Optional

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import ITEM_COVER, epub
from PIL import Image, UnidentifiedImageError


def extract_epub_text(epub_path: str, cache_path: str) -> str:
    logging.info(
        "extract_epub_text: epub_path=%s, cache_path=%s", epub_path, cache_path
    )
    md_cache_path = os.path.splitext(cache_path)[0] + ".md"
    if os.path.exists(md_cache_path):
        logging.info("extract_epub_text: cache hit %s", md_cache_path)
        with open(md_cache_path, "r", encoding="utf-8") as f:
            return f.read()
    logging.info("extract_epub_text: cache miss, extracting from %s", epub_path)
    book = epub.read_epub(epub_path)
    md_body_lines: List[str] = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
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
    # メタデータ抽出
    meta = extract_epub_metadata(epub_path)
    # YAML frontmatter
    md_lines = []
    md_lines.append("---")
    for k, v in meta.items():
        if v:
            md_lines.append(f"{k}: {v}")
    md_lines.append("---\n")
    if meta["title"]:
        md_lines.append(f"# {meta['title']}")
    md_lines.extend(md_body_lines)
    md_content = "\n".join(md_lines)
    with open(md_cache_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logging.info("extract_epub_text: cache written %s", md_cache_path)
    return md_content


def extract_epub_metadata(epub_path: str) -> dict[str, str]:
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


def get_epub_cover_path(epub_path: str, cache_dir: str) -> Optional[str]:
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


def extract_and_save_cover(epub_path: str, cover_path: str) -> Optional[str]:
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
