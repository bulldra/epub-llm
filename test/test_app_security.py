from fastapi.testclient import TestClient

from src.app import _is_safe_book_id, app

client = TestClient(app)


def test_is_safe_book_id_accepts_basename() -> None:
    assert _is_safe_book_id("book.epub") is True


def test_is_safe_book_id_rejects_traversal_and_separators() -> None:
    assert _is_safe_book_id("../evil.epub") is False
    assert _is_safe_book_id("a/evil.epub") is False
    assert _is_safe_book_id("..book.epub") is False


def test_download_invalid_book_id_returns_400() -> None:
    resp = client.get("/download/..book.epub")
    assert resp.status_code == 400
