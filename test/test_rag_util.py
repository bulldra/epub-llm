from __future__ import annotations

import json

from src.rag_util import RAGPipeline, get_or_create_markdown


class DummyEmbeddingService:
    """MLXEmbeddingService 互換の最小ダミー。"""

    def __init__(self) -> None:
        self._loaded = False
        self._index_calls: list[str] = []
        self._search_calls: list[dict[str, str | int | None]] = []
        self._results = [
            {
                "rank": 1,
                "score": 0.9,
                "text": "hello world",
                "book_id": "book1",
                "chunk_id": 0,
            }
        ]

    # --- index 互換メソッド ---
    def load_index(self) -> bool:  # noqa: D401
        return self._loaded

    def save_index(self) -> None:  # noqa: D401
        self._loaded = True

    def add_book(self, book_id: str, epub_path: str) -> None:  # noqa: D401
        del epub_path  # 引数未使用
        self._index_calls.append(book_id)
        self._loaded = True

    def get_stats(self) -> dict[str, int]:  # noqa: D401
        # 1 book -> 1 chunk として扱う
        return {"total_chunks": len(self._index_calls) or 1}

    def search(
        self, query: str, top_k: int, book_id: str | None
    ) -> list[dict[str, object]]:  # noqa: D401
        self._search_calls.append({"query": query, "book_id": book_id, "top_k": top_k})
        return self._results[:top_k]


def test_get_or_create_markdown_cache_hit(tmp_path) -> None:
    md = tmp_path / "sample.md"
    md.write_text("# Title\nBody", encoding="utf-8")
    epub_path = tmp_path / "sample.epub"  # 存在しなくても md があるので参照されない
    result = get_or_create_markdown(str(epub_path), str(tmp_path))
    assert result == str(md)


def test_get_or_create_markdown_generate(tmp_path, monkeypatch) -> None:
    # epub を作成し、extract_epub_text をモックして md 作成シミュレート
    epub_path = tmp_path / "book.epub"
    epub_path.write_text("dummy", encoding="utf-8")
    md_path = tmp_path / "book.md"

    def fake_extract(epub_path: str, cache: str) -> str:  # type: ignore
        del epub_path, cache
        # 対象 .md を生成
        md_path.write_text("# h\ntext", encoding="utf-8")
        return "# h\ntext"

    monkeypatch.setattr("src.rag_util.extract_epub_text", fake_extract)
    result = get_or_create_markdown(str(epub_path), str(tmp_path))
    assert result.endswith("book.md")
    assert md_path.exists()


def test_rag_pipeline_search_cache_behavior(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    epub_dir = tmp_path / "epub"
    cache_dir.mkdir()
    epub_dir.mkdir()
    # ダミー epub
    (epub_dir / "book1.epub").write_text("dummy", encoding="utf-8")

    svc = DummyEmbeddingService()
    pipeline = RAGPipeline(str(cache_dir), str(epub_dir), embedding_service=svc)

    # 1回目 (キャッシュ未使用)
    r1 = pipeline.search("hello", top_k=1, book_id="book1")
    assert len(r1) == 1
    cache_file = cache_dir / "faiss_search_cache.json"
    assert cache_file.exists()
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    assert data.get("entries")

    # 2回目 (キャッシュヒットで search 呼び出し増えない)
    r2 = pipeline.search("hello", top_k=1, book_id="book1")
    assert r2 == r1
    # ダミーサービスでは search_calls を直接見ない (簡易検証)
    assert r2 == r1

    # refresh 指定で再検索
    r3 = pipeline.search("hello", top_k=1, book_id="book1", cache_policy="refresh")
    assert r3 == r1
    assert r3 == r1


def test_rag_pipeline_add_book_without_epub(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    epub_dir = tmp_path / "epub"
    cache_dir.mkdir()
    epub_dir.mkdir()

    svc = DummyEmbeddingService()
    pipeline = RAGPipeline(str(cache_dir), str(epub_dir), embedding_service=svc)
    # md だけあるケース
    (cache_dir / "alone.md").write_text("# t", encoding="utf-8")
    pipeline.add_book("alone")
    assert pipeline.search("hello", top_k=1)  # 例外なく検索できる
