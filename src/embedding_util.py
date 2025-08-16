"""Embedding / FAISS ユーティリティ (テスト用最小実装).

主目的:
 - テキスト群をベクトル化 (開発/テスト時は疑似決定論ベクトル)
 - FAISS インデックス作成 / 検索
 - 埋め込み/テキストの永続化 (npy + json)
 - RAG 用簡易コンテキスト組み立て

本ファイルは test_embedding_util*.py の要求を満たす最小 API を提供する。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import faiss
import numpy as np

EMBED_DIM_DEV = 1024


@dataclass(frozen=True)
class ModelPair:
    """モデルとトークナイザの組 (実モデル無い場合は None)。"""

    model: Any | None
    tokenizer: Any | None


def _hash_to_vec(text: str, dim: int) -> np.ndarray:
    """テキストをハッシュし擬似決定論ベクトルへ。"""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    raw = (h * ((dim // len(h)) + 1))[:dim]
    arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    arr /= np.linalg.norm(arr) + 1e-9
    return arr


def create_embeddings_from_texts(
    texts: Sequence[str], model: Any | None, tokenizer: Any | None
) -> np.ndarray:
    """テキスト集合を埋め込みベクトルへ。

    model / tokenizer が None の場合、決定論ハッシュベクトルを生成。
    実モデル利用パスはテストでは未使用 (簡易実装)。
    """
    del tokenizer  # 現状未使用
    if model is None:
        vecs = [_hash_to_vec(t, EMBED_DIM_DEV) for t in texts]
        return np.vstack(vecs).astype(np.float32)
    raise RuntimeError("実モデル埋め込みは未実装です (テスト目的)。")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """FAISS インデックスを構築 (1D 可)。"""
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 1D or 2D array")
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    embeddings = embeddings / norms
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_embeddings(
    embeddings: np.ndarray, texts: Sequence[str], base_path: str
) -> None:
    np.save(base_path + ".npy", embeddings.astype(np.float32))
    with open(base_path + ".json", "w", encoding="utf-8") as f:
        json.dump(list(texts), f, ensure_ascii=False)


def load_embeddings(base_path: str) -> tuple[np.ndarray, list[str]]:
    emb = np.load(base_path + ".npy").astype(np.float32)
    with open(base_path + ".json", encoding="utf-8") as f:
        texts = json.load(f)
    if not isinstance(texts, list):
        raise ValueError("invalid texts json")
    return emb, [str(t) for t in texts]


def embed_texts_and_save(
    texts: Sequence[str], base_path: str, model: Any | None, tokenizer: Any | None
) -> np.ndarray:
    emb = create_embeddings_from_texts(texts, model, tokenizer)
    save_embeddings(emb, texts, base_path)
    return emb


def _encode_query(query: str, pair: ModelPair, dim: int) -> np.ndarray:
    """Encode query into an embedding vector.

    - If model is None: use deterministic hash vector of target ``dim``.
    - If model is provided: try common paths used by test doubles
      (``text_embeds`` or mean-pooled ``last_hidden_state``).
    """
    if pair.model is None:
        return _hash_to_vec("__q__" + query, dim).reshape(1, -1)
    # Tokenize single query (tests provide simple numpy outputs)
    if pair.tokenizer is None:
        raise RuntimeError("tokenizer is required when model is provided")
    tok = pair.tokenizer.batch_encode_plus([query])
    input_ids = tok.get("input_ids")
    attention_mask = tok.get("attention_mask")
    out = pair.model(input_ids, attention_mask)
    # Path 1: model returns text_embeds (1, d)
    vec = getattr(out, "text_embeds", None)
    if vec is not None:
        arr = np.asarray(vec, dtype=np.float32)
        reshaped = arr.reshape(arr.shape[0], -1)
        return cast(np.ndarray, reshaped)
    # Path 2: model returns last_hidden_state (b, seq_len, d) -> mean pool
    last = getattr(out, "last_hidden_state", None)
    if last is not None:
        arr = np.asarray(last, dtype=np.float32)
        if arr.ndim != 3:
            raise RuntimeError("invalid last_hidden_state shape")
        pooled = arr.mean(axis=1, keepdims=False)
        reshaped = pooled.reshape(pooled.shape[0], -1)
        return cast(np.ndarray, reshaped)
    raise RuntimeError(
        "unsupported model output (expected text_embeds or last_hidden_state)"
    )


def search_similar(
    query: str,
    pair: ModelPair,
    index: faiss.Index,
    texts: Sequence[str],
    top_k: int = 5,
) -> list[tuple[int, float, str]]:
    if not isinstance(index, faiss.Index):
        raise TypeError("index must be FAISS Index")
    top_k = max(1, min(top_k, len(texts)))
    dim = index.d
    qv = _encode_query(query, pair, dim)
    if qv.shape[1] != dim:
        raise ValueError("query embedding dimension mismatch")
    qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-9)
    scores, idx = index.search(qv, top_k)
    res: list[tuple[int, float, str]] = []
    for rank, (i, sc) in enumerate(zip(idx[0], scores[0], strict=False), start=1):
        if 0 <= i < len(texts):
            res.append((rank, float(sc), texts[i]))
    return res


def create_context_from_query(
    query: str,
    model_pair: ModelPair,
    index: faiss.Index,
    texts: Sequence[str],
    top_k: int = 3,
    join_delim: str = "\n------\n",
) -> str:
    hits = search_similar(query, model_pair, index, texts, top_k=top_k)
    return join_delim.join(h[2] for h in hits)


def load_and_search(
    query: str,
    base_path: str,
    model_pair: ModelPair,
    top_k: int = 5,
) -> list[tuple[int, float, str]]:
    emb, texts = load_embeddings(base_path)
    index = build_faiss_index(emb)
    return search_similar(query, model_pair, index, texts, top_k=top_k)


def iter_batch(it: Iterable[str], batch_size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for t in it:
        batch.append(t)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


__all__ = [
    "ModelPair",
    "create_embeddings_from_texts",
    "build_faiss_index",
    "save_embeddings",
    "load_embeddings",
    "embed_texts_and_save",
    "search_similar",
    "create_context_from_query",
    "load_and_search",
    "iter_batch",
]
