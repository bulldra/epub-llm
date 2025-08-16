from pathlib import Path

import pytest

from src.mlx_embedding_service import MLXEmbeddingService


def test_faiss_search_book_query_and_terms(tmp_path) -> None:
    epub_dir = Path("epub")
    assert epub_dir.is_dir(), "epub ディレクトリが存在しない"
    epub_files = sorted(epub_dir.glob("*.epub"))
    assert epub_files, "EPUB ファイルが存在しない"

    target = epub_files[0]
    book_id = target.name[:-5]
    queries = ["の", "Python", "メモリ", "Python 型"]

    model_name = "mlx-community/multilingual-e5-small-mlx"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    print("model_name:", model_name)
    svc = MLXEmbeddingService(cache_dir=str(cache_dir), model_name=model_name)
    try:
        svc.add_book(book_id, str(target))
    except RuntimeError as e:
        msg = str(e).lower()
        if "未サポート" in msg or "not supported" in msg:
            pytest.skip("モデル未対応: " + msg)
        raise

    stats = svc.get_stats()
    assert stats.get("total_books", 0) >= 1
    assert svc.chunks_metadata

    for q in queries:
        res = svc.search(q, top_k=3, book_id=book_id)
        assert isinstance(res, list)
        print("[FAISS-REAL] epub=", target.name)
        print("[FAISS-REAL] query=", q, "results=", len(res))
        if not res:
            continue
        top = res[0]
        assert "score" in top and "text" in top
        assert top["book_id"] == book_id
        assert top["score"] >= 0
