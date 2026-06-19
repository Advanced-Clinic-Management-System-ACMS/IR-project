"""
Pure mathematical scoring functions for IR retrieval.
No HTTP, database, or filesystem access — core domain logic only.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low, high = min(values), max(values)
    if high == low:
        return {doc_id: 1.0 for doc_id in scores}
    return {doc_id: (score - low) / (high - low) for doc_id, score in scores.items()}


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(vec_a.get(term, 0.0) * vec_b.get(term, 0.0) for term in set(vec_a) | set(vec_b))
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_query_tfidf_vector(
    query_tokens: list[str],
    idf: dict[str, float],
) -> dict[str, float]:
    if not query_tokens:
        return {}
    total = len(query_tokens)
    counts: dict[str, int] = {}
    for token in query_tokens:
        counts[token] = counts.get(token, 0) + 1
    return {term: (count / total) * idf.get(term, 0.0) for term, count in counts.items()}


def score_tfidf(
    query_tokens: list[str],
    tf_idf_vectors: dict[str, dict[str, float]],
    idf: dict[str, float],
) -> dict[str, float]:
    query_vector = build_query_tfidf_vector(query_tokens, idf)
    return {
        doc_id: cosine_similarity(query_vector, doc_vector)
        for doc_id, doc_vector in tf_idf_vectors.items()
    }


def score_bm25(
    query_tokens: list[str],
    inverted_index: dict[str, dict[str, int]],
    doc_lengths: dict[str, int],
    avg_doc_length: float,
    idf: dict[str, float],
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, float]:
    scores: dict[str, float] = {doc_id: 0.0 for doc_id in doc_lengths}
    for token in query_tokens:
        if token not in inverted_index:
            continue
        token_idf = idf.get(token, 0.0)
        for doc_id, tf in inverted_index[token].items():
            doc_len = doc_lengths.get(doc_id, 0)
            denom = tf + k1 * (1.0 - b + b * (doc_len / (avg_doc_length or 1.0)))
            scores[doc_id] += token_idf * ((tf * (k1 + 1.0)) / (denom or 1.0))
    return scores


def build_query_embedding_vector(
    query_tokens: list[str],
    vocabulary: list[str],
) -> np.ndarray:
    term_to_index = {term: idx for idx, term in enumerate(vocabulary)}
    vector = np.zeros(len(vocabulary), dtype=np.float32)
    if not query_tokens:
        return vector
    counts: dict[str, int] = {}
    for token in query_tokens:
        counts[token] = counts.get(token, 0) + 1
    for term, count in counts.items():
        idx = term_to_index.get(term)
        if idx is not None:
            vector[idx] = count / len(query_tokens)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


_FAISS_INDEX_CACHE = {}

def _get_faiss_index(embeddings: np.ndarray, cache_key: int):
    """Retrieves or builds a highly optimized FAISS vector store index."""
    import faiss
    if cache_key not in _FAISS_INDEX_CACHE:
        dimension = embeddings.shape[1]
        # IndexFlatIP uses Inner Product (which equals Cosine Similarity for normalized vectors)
        vector_store = faiss.IndexFlatIP(dimension)
        # FAISS requires contiguous float32 arrays
        vector_store.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        _FAISS_INDEX_CACHE[cache_key] = vector_store
    return _FAISS_INDEX_CACHE[cache_key]
# ---------------------------------------


def score_embedding(
    query_tokens: list[str],
    embeddings: np.ndarray,
    doc_ids: list[str],
    vocabulary: list[str],
) -> dict[str, float]:
    if embeddings.size == 0 or not doc_ids:
        return {}
        
    query_vector = build_query_embedding_vector(query_tokens, vocabulary)
    q_vec_2d = np.array([query_vector], dtype=np.float32)

    # Use FAISS Vector Store instead of raw Numpy
    cache_key = id(embeddings)
    vector_store = _get_faiss_index(embeddings, cache_key)
    
    k = min(100, len(doc_ids))
    scores, indices = vector_store.search(q_vec_2d, k)

    results = {}
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results[doc_ids[idx]] = float(score)
            
    return results


def score_embedding_semantic(
    query_tokens: list[str],
    query_text: str,
    embeddings: np.ndarray,
    doc_ids: list[str],
    model_name: str,
) -> dict[str, float]:
    if embeddings.size == 0 or not doc_ids:
        return {}
        
    from retrieval_service.infrastructure.embedding_model import encode_text

    text = query_text.strip() or " ".join(query_tokens)
    query_vector = encode_text(text, model_name)
    q_vec_2d = np.array([query_vector], dtype=np.float32)

    # Use FAISS Vector Store instead of raw Numpy
    cache_key = id(embeddings)
    vector_store = _get_faiss_index(embeddings, cache_key)
    
    k = min(100, len(doc_ids))
    scores, indices = vector_store.search(q_vec_2d, k)

    results = {}
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results[doc_ids[idx]] = float(score)
            
    return results


def _embedding_scores(
    query_tokens: list[str],
    index_data: dict,
) -> dict[str, float]:
    if index_data.get("embeddings") is None or not index_data.get("embedding_doc_ids"):
        return {}
    if index_data.get("embedding_type") == "sentence_transformer":
        return score_embedding_semantic(
            query_tokens,
            index_data.get("query_text", ""),
            index_data["embeddings"],
            index_data["embedding_doc_ids"],
            index_data["embedding_model"],
        )
    return score_embedding(
        query_tokens,
        index_data["embeddings"],
        index_data["embedding_doc_ids"],
        index_data["vocabulary"],
    )


def fuse_serial_rerank(
    query_tokens: list[str],
    index_data: dict,
    k1: float,
    b: float,
    candidate_k: int = 100,
) -> dict[str, float]:
    """Serial hybrid: BM25 retrieves candidates, embeddings re-rank them."""
    bm25_scores = score_bm25(
        query_tokens,
        index_data["inverted_index"],
        index_data["doc_lengths"],
        index_data["avg_doc_length"],
        index_data["idf"],
        k1=k1,
        b=b,
    )
    candidates = rank_documents(bm25_scores, candidate_k)
    candidate_ids = [doc_id for doc_id, _ in candidates]

    if index_data.get("embeddings") is None or not index_data.get("embedding_doc_ids"):
        return dict(candidates)

    embedding_scores = _embedding_scores(query_tokens, index_data)
    return {doc_id: embedding_scores.get(doc_id, 0.0) for doc_id in candidate_ids}


def fuse_serial(
    tfidf_scores: dict[str, float],
    bm25_scores: dict[str, float],
    embedding_scores: dict[str, float],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    weights = weights or {"tfidf": 0.34, "bm25": 0.33, "embedding": 0.33}
    norm_tfidf = _min_max_normalize(tfidf_scores)
    norm_bm25 = _min_max_normalize(bm25_scores)
    norm_embed = _min_max_normalize(embedding_scores)
    all_docs = set(norm_tfidf) | set(norm_bm25) | set(norm_embed)
    return {
        doc_id: (
            weights.get("tfidf", 0.34) * norm_tfidf.get(doc_id, 0.0)
            + weights.get("bm25", 0.33) * norm_bm25.get(doc_id, 0.0)
            + weights.get("embedding", 0.33) * norm_embed.get(doc_id, 0.0)
        )
        for doc_id in all_docs
    }


def fuse_parallel(
    tfidf_scores: dict[str, float],
    bm25_scores: dict[str, float],
    embedding_scores: dict[str, float],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    return fuse_serial(tfidf_scores, bm25_scores, embedding_scores, weights)


def fuse_rrf(
    score_maps: list[dict[str, float]],
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal Rank Fusion across multiple ranked lists."""
    fused: dict[str, float] = {}
    for scores in score_maps:
        if not scores:
            continue
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


def fuse_branching(
    query_tokens: list[str],
    tfidf_scores: dict[str, float],
    bm25_scores: dict[str, float],
    embedding_scores: dict[str, float],
) -> dict[str, float]:
    token_count = len(query_tokens)
    if token_count <= 2:
        return bm25_scores
    if token_count <= 5:
        return tfidf_scores
    return embedding_scores


def rank_documents(scores: dict[str, float], top_k: int) -> list[tuple[str, float]]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


def compute_all_lexical_scores(
    query_tokens: list[str],
    index_data: dict,
    k1: float,
    b: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    tfidf_scores = score_tfidf(query_tokens, index_data["tf_idf_vectors"], index_data["idf"])
    bm25_scores = score_bm25(
        query_tokens,
        index_data["inverted_index"],
        index_data["doc_lengths"],
        index_data["avg_doc_length"],
        index_data["idf"],
        k1=k1,
        b=b,
    )
    embedding_scores = _embedding_scores(query_tokens, index_data)
    return tfidf_scores, bm25_scores, embedding_scores


def run_parallel_scorers(
    query_tokens: list[str],
    index_data: dict,
    k1: float,
    b: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    def score_tfidf_task() -> dict[str, float]:
        return score_tfidf(query_tokens, index_data["tf_idf_vectors"], index_data["idf"])

    def score_bm25_task() -> dict[str, float]:
        return score_bm25(
            query_tokens,
            index_data["inverted_index"],
            index_data["doc_lengths"],
            index_data["avg_doc_length"],
            index_data["idf"],
            k1=k1,
            b=b,
        )

    def score_embedding_task() -> dict[str, float]:
        return _embedding_scores(query_tokens, index_data)

    with ThreadPoolExecutor(max_workers=3) as executor:
        tfidf_future = executor.submit(score_tfidf_task)
        bm25_future = executor.submit(score_bm25_task)
        embed_future = executor.submit(score_embedding_task)
        return tfidf_future.result(), bm25_future.result(), embed_future.result()
