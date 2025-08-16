from __future__ import annotations

import os
import platform
from pathlib import Path

import pytest

from src.mlx_embedding_service import MLXEmbeddingService


@pytest.mark.timeout(300)
def test_mlx_real_epub_end_to_end(tmp_path) -> None:  # noqa: D401
    """実際の EPUB 1冊で add_book -> stats -> search を検証 (任意)。

    実行条件:
      - 環境変数 RUN_MLX_E2E=1
      - Apple Silicon macOS
      - ネットワーク利用可能

    条件を満たさない場合 / モデル未準備の場合は skip。
    """

    if os.environ.get("RUN_MLX_E2E") != "1":
        pytest.skip("RUN_MLX_E2E=1 で有効化")
    if platform.system() != "Darwin" or platform.machine() not in {"arm64", "arm"}:
        pytest.skip("Apple Silicon macOS 以外はスキップ")

    model_name = os.environ.get(
        "MLX_E2E_MODEL", "mlx-community/multilingual-e5-small-mlx"
    )

    epub_dir = Path("epub")
    if not epub_dir.is_dir():
        pytest.skip("epub ディレクトリが存在しない")
    epub_files = sorted(epub_dir.glob("*.epub"))
    if not epub_files:
        pytest.skip("EPUB ファイルが存在しない")
    target_epub = epub_files[0]
    book_id = target_epub.name[:-5]

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    svc = MLXEmbeddingService(cache_dir=str(cache_dir), model_name=model_name)

    try:
        svc.add_book(book_id, str(target_epub))
    except RuntimeError as e:
        msg = str(e)
        if any(
            key in msg.lower()
            for key in [
                "safetensors",
                "mlx ランタイム",
                "ロードできません",
                "cache削除",
            ]
        ):
            pytest.skip(f"モデル未準備のためスキップ: {msg[:120]}")
        raise

    stats = svc.get_stats()
    assert stats.get("total_books", 0) >= 1
    assert stats.get("index_dimension", 0) > 0

    results = svc.search("の", top_k=1, book_id=book_id)
    assert results, "検索結果が空"
    assert "score" in results[0]
    assert results[0]["book_id"] == book_id
