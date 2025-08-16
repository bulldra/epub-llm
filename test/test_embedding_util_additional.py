from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from src.embedding_util import (
    ModelPair,
    build_faiss_index,
    create_context_from_query,
    embed_texts_and_save,
    load_embeddings,
    save_embeddings,
    search_similar,
)


def test_build_faiss_index_reshapes_1d_vector() -> None:
    vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    index = build_faiss_index(vec)
    assert index.d == 4


def test_build_faiss_index_invalid_dimension_raises() -> None:
    bad = np.zeros((2, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        build_faiss_index(bad)


def test_save_and_load_embeddings_roundtrip() -> None:
    texts = ["t1", "t2", "t3"]
    emb = np.random.random((3, 8)).astype(np.float32)
    tmpdir = tempfile.mkdtemp(prefix="emb-roundtrip-")
    base = os.path.join(tmpdir, "embs")
    save_embeddings(emb, texts, base)
    loaded_emb, loaded_texts = load_embeddings(base)
    assert loaded_texts == texts
    assert loaded_emb.shape == emb.shape
    assert np.allclose(loaded_emb, emb)


def test_embed_texts_and_save_files_created() -> None:
    texts = ["alpha", "beta"]
    tmpdir = tempfile.mkdtemp(prefix="emb-save-")
    base = os.path.join(tmpdir, "embs")
    embed_texts_and_save(texts, base, model=None, tokenizer=None)
    assert os.path.exists(base + ".npy")
    assert os.path.exists(base + ".json")
    emb, loaded_texts = load_embeddings(base)
    assert loaded_texts == texts
    assert emb.shape[0] == len(texts)


def test_create_context_from_query_custom_delimiter() -> None:
    texts = ["one", "two", "three"]
    emb = np.random.random((3, 16)).astype(np.float32)
    index = build_faiss_index(emb)
    pair = ModelPair(model=None, tokenizer=None)
    delim = "\n***\n"
    ctx = create_context_from_query(
        "dummy", pair, index, texts, top_k=2, join_delim=delim
    )
    assert ctx.count("***") == 1


def test_search_similar_dimension_mismatch_raises() -> None:
    emb = np.random.random((5, 4)).astype(np.float32)
    index = build_faiss_index(emb)
    texts = [f"t{i}" for i in range(5)]

    class DummyTokenizer:
        def batch_encode_plus(self, batch, **_kwargs):
            ids = np.zeros((len(batch), 3), dtype=np.int64)
            mask = np.ones_like(ids, dtype=np.int64)
            return {"input_ids": ids, "attention_mask": mask}

    class DummyModel:
        def __call__(self, input_ids, attention_mask):
            return SimpleNamespace(
                text_embeds=np.random.random((1, 3)).astype(np.float32)
            )

    pair = ModelPair(model=DummyModel(), tokenizer=DummyTokenizer())
    with pytest.raises(ValueError):
        search_similar("q", pair, index, texts, top_k=2)


@dataclass
class _DummyModelLastHidden:
    dim: int

    def __call__(self, input_ids, attention_mask):
        b, seq_len = input_ids.shape
        return SimpleNamespace(
            last_hidden_state=np.random.random((b, seq_len, self.dim))
        )


def test_search_similar_last_hidden_state_path() -> None:
    texts = ["aa", "bb", "cc"]
    emb = np.random.random((3, 6)).astype(np.float32)
    index = build_faiss_index(emb)

    class DummyTokenizer2:
        def batch_encode_plus(self, batch, **_kwargs):
            ids = np.zeros((len(batch), 4), dtype=np.int64)
            mask = np.ones_like(ids, dtype=np.int64)
            return {"input_ids": ids, "attention_mask": mask}

    model = _DummyModelLastHidden(dim=6)
    pair = ModelPair(model=model, tokenizer=DummyTokenizer2())
    results = search_similar("query", pair, index, texts, top_k=2)
    assert len(results) == 2
    assert all(r[2] in texts for r in results)
