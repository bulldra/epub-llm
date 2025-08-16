from __future__ import annotations

import argparse
import logging
import os

from src.config_manager import AppConfig
from src.mlx_embedding_service import MLXEmbeddingService


def build(epub_dir: str | None = None, cache_dir: str | None = None) -> None:
    cfg = AppConfig()
    epub_dir = epub_dir or cfg.get("directories.epub_dir")
    cache_dir = cache_dir or cfg.get("directories.cache_dir")
    if not isinstance(epub_dir, str) or not epub_dir:
        raise SystemExit("epub_dir が取得できません")
    if not isinstance(cache_dir, str) or not cache_dir:
        raise SystemExit("cache_dir が取得できません")
    if not os.path.isdir(epub_dir):
        raise SystemExit(f"EPUB ディレクトリが存在しません: {epub_dir}")
    svc = MLXEmbeddingService(cache_dir=cache_dir)
    svc.build_index(epub_dir)
    stats = svc.get_stats()
    logging.info("Index built: %s", stats)


def search(query: str, book_id: str | None, top_k: int) -> None:
    cfg = AppConfig()
    cache_dir = cfg.get("directories.cache_dir")
    if not isinstance(cache_dir, str):
        raise SystemExit("cache_dir 未設定")
    svc = MLXEmbeddingService(cache_dir=cache_dir)
    # 既存インデックス読み込みできなければ再構築
    if not svc.load_index():
        epub_dir = cfg.get("directories.epub_dir")
        if not isinstance(epub_dir, str):
            raise SystemExit("epub_dir 未設定")
        svc.build_index(epub_dir)
    results = svc.search(query=query, top_k=top_k, book_id=book_id)
    for r in results:
        if "error" in r:
            print(r["error"])  # noqa: T201
            continue
        print(f"[{r.get('score'):.3f}] {r.get('book_id')} #{r.get('chunk_id')}:")  # noqa: T201
        text = str(r.get("text", ""))
        print(text[:240].replace("\n", " ") + ("..." if len(text) > 240 else ""))  # noqa: T201
        print("-")  # noqa: T201


def main() -> None:  # noqa: D401
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description="MLX Embedding index builder/searcher")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="EPUB から新規にインデックスを構築")
    b.add_argument(
        "--epub-dir", dest="epub_dir", help="EPUB ディレクトリ", default=None
    )
    b.add_argument(
        "--cache-dir", dest="cache_dir", help="キャッシュディレクトリ", default=None
    )

    s = sub.add_parser("search", help="既存インデックスを使って検索")
    s.add_argument("query", help="検索クエリ")
    s.add_argument(
        "--book-id", dest="book_id", default=None, help="対象書籍ID (拡張子抜き)"
    )
    s.add_argument("--top-k", dest="top_k", type=int, default=5)

    args = p.parse_args()
    if args.cmd == "build":
        build(args.epub_dir, args.cache_dir)
    elif args.cmd == "search":
        search(args.query, args.book_id, args.top_k)
    else:  # pragma: no cover
        p.error("不明なコマンド")


if __name__ == "__main__":  # pragma: no cover
    main()
