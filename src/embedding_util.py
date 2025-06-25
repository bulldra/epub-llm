import json
import logging
from typing import Any, List, Tuple

import faiss
import numpy as np
from tqdm import tqdm


def create_embeddings_from_texts(
    texts: List[str],
    model: Any,
    tokenizer: Any,
    batch_size: int = 32,
) -> np.ndarray:
    embeddings: List[np.ndarray] = []
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
        embeddings.append(emb)
    return np.concatenate(embeddings, axis=0)


def save_embeddings(embeddings: np.ndarray, texts: List[str], out_path: str) -> None:
    np.save(out_path + ".npy", embeddings)
    with open(out_path + ".json", "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)


def load_embeddings(
    base_path: str,
) -> Tuple[np.ndarray, List[str]]:
    embeddings = np.load(base_path + ".npy")
    with open(base_path + ".json", "r", encoding="utf-8") as f:
        texts = json.load(f)
    return embeddings, texts


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index


def search_similar(
    query: str,
    model: Any,
    tokenizer: Any,
    index: faiss.IndexFlatL2,
    texts: List[str],
    top_k: int = 5,
) -> List[Tuple[int, float, str]]:
    inputs = tokenizer.batch_encode_plus(
        [query],
        return_tensors="mlx",
        padding=True,
        truncation=True,
        max_length=512,
    )
    outputs = model(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )
    query_emb = outputs.text_embeds
    query_emb = np.asarray(query_emb, dtype=np.float32)
    distances, indices = index.search(query_emb, top_k)
    results: List[Tuple[int, float, str]] = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append((idx, dist, texts[idx]))
    return results


def create_context_from_query(
    query: str,
    model: Any,
    tokenizer: Any,
    index: faiss.IndexFlatL2,
    texts: List[str],
    top_k: int = 5,
    join_delim: str = "\n\n------\n\n",
) -> str:
    results: List[Tuple[int, float, str]] = search_similar(
        query=query,
        model=model,
        tokenizer=tokenizer,
        index=index,
        texts=texts,
        top_k=top_k,
    )
    context: str = join_delim.join([r[2] for r in results])
    return context


def embed_texts_and_save(
    texts: List[str],
    out_path: str,
    model: Any,
    tokenizer: Any,
    batch_size: int = 32,
) -> None:
    logging.info("Embedding %d texts and saving to %s", len(texts), out_path)
    embeddings = create_embeddings_from_texts(
        texts=texts, model=model, tokenizer=tokenizer, batch_size=batch_size
    )
    save_embeddings(embeddings, texts, out_path)


def load_and_search(
    query: str,
    base_path: str,
    model: Any,
    tokenizer: Any,
    top_k: int = 5,
) -> List[Tuple[int, float, str]]:
    logging.info(
        "Loading embeddings from %s and searching for query: %s",
        base_path,
        query,
    )
    embeddings, texts = load_embeddings(base_path)
    index = build_faiss_index(embeddings)
    return search_similar(query, model, tokenizer, index, texts, top_k=top_k)
