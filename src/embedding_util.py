"""Embedding utilities for text processing and similarity search.

This module provides functions for creating embeddings from text,
saving/loading embeddings, building FAISS indices, and performing
semantic similarity searches.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
from tqdm import tqdm


@dataclass
class ModelPair:
    """Container for model and tokenizer pair."""

    model: Any
    tokenizer: Any


def create_embeddings_from_texts(
    texts: list[str],
    model: Any,
    tokenizer: Any,
    batch_size: int = 32,
) -> np.ndarray:
    """Create embeddings from a list of texts using a pre-trained model.

    Args:
        texts: List of input texts to embed.
        model: The embedding model to use.
        tokenizer: The tokenizer for the model.
        batch_size: Number of texts to process in each batch.

    Returns:
        A numpy array of embeddings with shape (num_texts, embedding_dim).
    """
    # Development mode: return mock embeddings
    if model is None or tokenizer is None:
        logging.warning("Development mode: generating mock embeddings")
        # Create mock embeddings with standard dimension (1024)
        num_texts = len(texts)
        mock_embeddings = np.random.normal(0, 1, (num_texts, 1024)).astype(np.float32)
        return mock_embeddings

    embeddings: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer.batch_encode_plus(
            batch,
            return_tensors="mlx",
            padding=True,
            truncation=True,
            max_length=512,
        )
        outputs = model(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        emb = outputs.text_embeds
        # Ensure embeddings are properly shaped
        emb = np.asarray(emb, dtype=np.float32)
        embeddings.append(emb)
    return np.concatenate(embeddings, axis=0)


def save_embeddings(embeddings: np.ndarray, texts: list[str], out_path: str) -> None:
    """Save embeddings and texts to disk.

    Args:
        embeddings: Numpy array of embeddings to save.
        texts: List of corresponding texts.
        out_path: Base path for saving (will create .npy and .json files).
    """
    np.save(out_path + ".npy", embeddings)
    with open(out_path + ".json", "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)


def load_embeddings(
    base_path: str,
) -> tuple[np.ndarray, list[str]]:
    """Load embeddings and texts from disk.

    Args:
        base_path: Base path to load from (expects .npy and .json files).

    Returns:
        Tuple of (embeddings array, list of texts).
    """
    embeddings = np.load(base_path + ".npy")
    with open(base_path + ".json", encoding="utf-8") as f:
        texts = json.load(f)
    return embeddings, texts


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build a FAISS index from embeddings for similarity search.

    Args:
        embeddings: Numpy array of embeddings.

    Returns:
        A FAISS IndexFlatL2 index ready for searching.

    Raises:
        ValueError: If embeddings are not 2D after reshaping.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    d = embeddings.shape[1]
    logging.info(
        "Building FAISS index with dimension: %d, shape: %s", d, embeddings.shape
    )
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index


def search_similar(
    query: str,
    model_pair: ModelPair,
    index: faiss.IndexFlatL2,
    texts: list[str],
    top_k: int = 5,
) -> list[tuple[int, float, str]]:
    """Search for similar texts using semantic similarity.

    Args:
        query: The query text to search for.
        model_pair: Container with embedding model and tokenizer.
        index: Pre-built FAISS index.
        texts: List of texts corresponding to the index.
        top_k: Number of similar texts to return.

    Returns:
        List of tuples (index, distance, text) for the most similar texts.
    """
    # Development mode: return mock results
    if model_pair.model is None or model_pair.tokenizer is None:
        logging.warning("Development mode: returning mock search results")
        # Return first few texts as mock results
        mock_results = []
        for i in range(min(top_k, len(texts))):
            mock_results.append((i, 0.5, texts[i]))
        return mock_results

    inputs = model_pair.tokenizer.batch_encode_plus(
        [query],
        return_tensors="mlx",
        padding=True,
        truncation=True,
        max_length=512,
    )
    outputs = model_pair.model(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )
    query_emb = outputs.text_embeds
    query_emb = np.asarray(query_emb, dtype=np.float32)

    # Ensure query embedding dimensions match index
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)

    logging.info(
        "Query embedding shape: %s, Index dimension: %d", query_emb.shape, index.d
    )

    if query_emb.shape[1] != index.d:
        raise ValueError(
            f"Query embedding dimension {query_emb.shape[1]} does not match "
            f"index dimension {index.d}"
        )

    distances, indices = index.search(query_emb, top_k)
    results: list[tuple[int, float, str]] = []
    for idx, dist in zip(indices[0], distances[0], strict=False):
        results.append((idx, dist, texts[idx]))
    return results


def create_context_from_query(
    query: str,
    model_pair: ModelPair,
    index: faiss.IndexFlatL2,
    texts: list[str],
    top_k: int = 5,
    *,
    join_delim: str = "\n\n------\n\n",
) -> str:
    """Create a context string from similar texts for a query.

    Args:
        query: The query text.
        model_pair: Container with embedding model and tokenizer.
        index: Pre-built FAISS index.
        texts: List of texts corresponding to the index.
        top_k: Number of similar texts to include.
        join_delim: Delimiter to join the similar texts.

    Returns:
        A string containing the joined similar texts.
    """
    results: list[tuple[int, float, str]] = search_similar(
        query=query,
        model_pair=model_pair,
        index=index,
        texts=texts,
        top_k=top_k,
    )
    context: str = join_delim.join([r[2] for r in results])
    return context


def embed_texts_and_save(
    texts: list[str],
    out_path: str,
    model: Any,
    tokenizer: Any,
    batch_size: int = 32,
) -> None:
    """Create embeddings from texts and save them to disk.

    Args:
        texts: List of texts to embed.
        out_path: Base path for saving the embeddings.
        model: The embedding model.
        tokenizer: The tokenizer for the model.
        batch_size: Number of texts to process in each batch.
    """
    logging.info("Embedding %d texts and saving to %s", len(texts), out_path)
    embeddings = create_embeddings_from_texts(
        texts=texts, model=model, tokenizer=tokenizer, batch_size=batch_size
    )
    save_embeddings(embeddings, texts, out_path)


def load_and_search(
    query: str,
    base_path: str,
    model_pair: ModelPair,
    top_k: int = 5,
) -> list[tuple[int, float, str]]:
    """Load embeddings from disk and search for similar texts.

    Args:
        query: The query text to search for.
        base_path: Base path to load embeddings from.
        model_pair: Container with embedding model and tokenizer.
        top_k: Number of similar texts to return.

    Returns:
        List of tuples (index, distance, text) for the most similar texts.
    """
    logging.info(
        "Loading embeddings from %s and searching for query: %s",
        base_path,
        query,
    )
    embeddings, texts = load_embeddings(base_path)
    index = build_faiss_index(embeddings)
    return search_similar(query, model_pair, index, texts, top_k=top_k)
