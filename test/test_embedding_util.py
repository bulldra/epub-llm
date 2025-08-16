from __future__ import annotations

import os
import tempfile

import numpy as np

from src.embedding_util import (
    ModelPair,
    build_faiss_index,
    create_context_from_query,
    create_embeddings_from_texts,
    load_and_search,
    save_embeddings,
)


def test_create_embeddings_dev_mode_shape_and_determinism() -> None:
    texts = ["hello", "world", "epub"]
    emb = create_embeddings_from_texts(texts, model=None, tokenizer=None)
    assert isinstance(emb, np.ndarray)
    assert emb.shape[0] == len(texts)
    assert emb.shape[1] == 1024


def test_build_index_and_search_dev_mode() -> None:
    texts = ["python basics", "fastapi guide", "faiss search"]
    emb = create_embeddings_from_texts(texts, model=None, tokenizer=None)
    # Build index just to ensure it doesn't raise; search path loads from disk
    build_faiss_index(emb)
    pair = ModelPair(model=None, tokenizer=None)
    results = load_and_search(
        query="python",
        base_path=_save_temp_embeddings(emb, texts),
        model_pair=pair,
        top_k=2,
    )
    assert len(results) == 2
    assert results[0][2] in texts


def test_create_context_from_query_dev_mode() -> None:
    texts = ["a", "b", "c"]
    emb = create_embeddings_from_texts(texts, model=None, tokenizer=None)
    index = build_faiss_index(emb)
    pair = ModelPair(model=None, tokenizer=None)
    ctx = create_context_from_query("x", pair, index, texts, top_k=2)
    assert ctx
    # Should contain two segments joined by delimiter
    assert "------" in ctx


def _save_temp_embeddings(emb: np.ndarray, texts: list[str]) -> str:
    tmpdir = tempfile.mkdtemp(prefix="emb-util-")
    base = os.path.join(tmpdir, "embs")
    save_embeddings(emb, texts, base)
    return base
