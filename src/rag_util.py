"""Lightweight RAG 管理ユーティリティ。

目的:
1. EPUB → Markdown 抽出 (既存キャッシュ再利用)
2. MLX / 任意埋め込みサービスでの FAISS インデックス再利用
3. 検索結果のシンプルキャッシュ (同一 query / book_id / top_k)

テストやスクリプトから簡易利用するための薄いラッパ。
本番アプリは既存の MLXEmbeddingService / FastAPI を優先利用。
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from src.epub_util import extract_epub_text

LOGGER = logging.getLogger(__name__)

SearchResult = dict[str, Any]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def get_or_create_markdown(epub_path: str, cache_dir: str) -> str:
    """EPUB から Markdown を生成 (既に存在すれば再利用) しパスを返す。

    epub_path が存在しない場合でも `<cache_dir>/<book_id>.md` があればそれを返す。
    戻り値: markdown ファイルパス
    失敗時: FileNotFoundError
    """
    book_id = os.path.splitext(os.path.basename(epub_path))[0]
    txt_cache = os.path.join(cache_dir, f"{book_id}.txt")
    md_cache = os.path.join(cache_dir, f"{book_id}.md")
    if os.path.exists(md_cache):
        return md_cache
    if not os.path.exists(epub_path):  # md 無く epub も無い
        raise FileNotFoundError(f"EPUB も Markdown も存在しません: {epub_path}")
    os.makedirs(cache_dir, exist_ok=True)
    # extract_epub_text は txt_cache を与えると .md を作成する実装
    extract_epub_text(epub_path, txt_cache)
    if not os.path.exists(md_cache):  # 念のため
        raise FileNotFoundError(f"Markdown 生成失敗: {md_cache}")
    return md_cache


@dataclass
class SearchCacheEntry:
    """検索結果キャッシュ 1 件分の保持構造。"""

    query: str
    book_id: str | None
    top_k: int
    results: list[SearchResult]
    created_at: str
    chunks_total: int


class RAGPipeline:
    """簡易 RAG パイプライン。

    埋め込みサービスは MLXEmbeddingService 互換 (add_book, search, save_index,
    load_index, get_stats) の任意オブジェクトを受け入れる。
    """

    def __init__(
        self,
        cache_dir: str,
        epub_dir: str,
        embedding_service: Any | None = None,
        model_loader: Callable[[], Any] | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.epub_dir = epub_dir
        os.makedirs(cache_dir, exist_ok=True)
        # 遅延 import (モデルロード回避のため)
        if embedding_service is None:
            from src.mlx_embedding_service import MLXEmbeddingService

            embedding_service = MLXEmbeddingService(cache_dir)
        self.embedding_service = embedding_service
        self._search_cache_path = os.path.join(cache_dir, "faiss_search_cache.json")
        self._search_cache: list[SearchCacheEntry] = []
        self._cache_loaded = False
        self._model_loader = model_loader  # 将来拡張用

    # --------- 内部ユーティリティ ---------
    def _load_search_cache(self) -> None:
        if self._cache_loaded:
            return
        if os.path.exists(self._search_cache_path):
            try:
                with open(self._search_cache_path, encoding="utf-8") as f:
                    raw = json.load(f)
                entries = raw.get("entries", []) if isinstance(raw, dict) else []
                for e in entries:
                    if not isinstance(e, dict):
                        continue
                    self._search_cache.append(
                        SearchCacheEntry(
                            query=str(e.get("query")),
                            book_id=e.get("book_id"),
                            top_k=int(e.get("top_k", 0)),
                            results=e.get("results", []),
                            created_at=str(e.get("created_at", "")),
                            chunks_total=int(e.get("chunks_total", 0)),
                        )
                    )
            except (OSError, ValueError, TypeError) as exc:
                LOGGER.warning("検索キャッシュ読込失敗: %s", exc)
        self._cache_loaded = True

    def _save_search_cache(self) -> None:
        try:
            data = {
                "updated_at": _now_iso(),
                "entries": [e.__dict__ for e in self._search_cache],
            }
            with open(self._search_cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except (OSError, ValueError, TypeError) as exc:
            LOGGER.warning("検索キャッシュ保存失敗: %s", exc)

    def _match_cache(
        self, query: str, book_id: str | None, top_k: int, chunks_total: int
    ) -> list[SearchResult] | None:
        self._load_search_cache()
        for entry in self._search_cache:
            if (
                entry.query == query
                and entry.book_id == book_id
                and entry.top_k == top_k
                and entry.chunks_total == chunks_total
            ):
                return entry.results
        return None

    def _store_cache(
        self,
        query: str,
        book_id: str | None,
        top_k: int,
        chunks_total: int,
        results: list[SearchResult],
    ) -> None:
        # 既存重複は追加しない
        if self._match_cache(query, book_id, top_k, chunks_total) is not None:
            return
        self._search_cache.append(
            SearchCacheEntry(
                query=query,
                book_id=book_id,
                top_k=top_k,
                results=results,
                created_at=_now_iso(),
                chunks_total=chunks_total,
            )
        )
        self._save_search_cache()

    # --------- 公開 API ---------
    def ensure_index(self) -> None:
        """インデックスをロード (無ければ構築)。"""
        if self.embedding_service.load_index():
            return
        # 無い場合は EPUB から構築
        epub_files = [f for f in os.listdir(self.epub_dir) if f.endswith(".epub")]
        for epub_file in epub_files:
            book_id = epub_file[:-5]
            full = os.path.join(self.epub_dir, epub_file)
            try:
                self.embedding_service.add_book(book_id, full)
            except FileNotFoundError:
                continue
        try:
            self.embedding_service.save_index()
        except (OSError, ValueError, RuntimeError) as exc:
            LOGGER.debug("save_index 失敗(無視): %s", exc)

    def add_book(self, book_id: str) -> None:
        """単一書籍をインデックス化 (既存 md 再利用)。

        - EPUB が存在しない場合でも cache_dir に <book_id>.md があれば追加可能
        - 既にインデックス化済みかどうかの判定はシンプル化 (再追加は FAISS が内部で扱う)
        """
        epub_path = os.path.join(self.epub_dir, f"{book_id}.epub")
        md_path = os.path.join(self.cache_dir, f"{book_id}.md")
        if not os.path.exists(epub_path) and not os.path.exists(md_path):
            raise FileNotFoundError(
                f"EPUB / Markdown が存在しません: {book_id} (期待: {epub_path} or {md_path})"
            )
        self.embedding_service.add_book(book_id, epub_path)
        try:
            self.embedding_service.save_index()
        except (OSError, ValueError, RuntimeError) as exc:
            LOGGER.debug("save_index 失敗(無視): %s", exc)

    def search(
        self,
        query: str,
        top_k: int = 5,
        book_id: str | None = None,
        use_cache: bool = True,
        cache_policy: Literal["prefer", "refresh", "ignore"] = "prefer",
    ) -> list[SearchResult]:
        """検索を実行し結果を返す。

        cache_policy:
            prefer  : キャッシュあれば利用 (デフォルト)
            refresh : 毎回再検索しキャッシュ更新
            ignore  : キャッシュ読み書き無し
        """
        self.ensure_index()
        stats = self.embedding_service.get_stats()
        chunks_total = (
            int(stats.get("total_chunks", 0)) if isinstance(stats, dict) else 0
        )
        if use_cache and cache_policy == "prefer":
            cached = self._match_cache(query, book_id, top_k, chunks_total)
            if cached is not None:
                return cached
        if cache_policy == "ignore":
            use_cache = False
        results = self.embedding_service.search(
            query=query, top_k=top_k, book_id=book_id
        )
        if use_cache and cache_policy != "ignore":
            self._store_cache(query, book_id, top_k, chunks_total, results)
        return results
