"""
Scoring using sklearn TF-IDF, rank_bm25 BM25, and pre-built offline FAISS index.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np


def score_tfidf(query_tokens: list[str], index_data: dict) -> dict[str, float]:
    if not query_tokens:
        return {}
    vectorizer = index_data["tfidf_vectorizer"]
    matrix = index_data["tfidf_matrix"]
    doc_ids = index_data["tfidf_doc_ids"]
    query_vec = vectorizer.transform([" ".join(query_tokens)])
    scores = (matrix @ query_vec.T).toarray().ravel()
    return {doc_ids[i]: float(scores[i]) for i in range(len(doc_ids))}


def score_bm25(
    query_tokens: list[str],
    index_data: dict,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, float]:
    if not query_tokens:
        return {}
    bm25 = index_data["bm25_model"]
    doc_ids = index_data["bm25_doc_ids"]
    bm25.k1 = k1
    bm25.b = b
    scores = bm25.get_scores(query_tokens)
    return {doc_ids[i]: float(scores[i]) for i in range(len(doc_ids))}


def score_embedding_semantic(
    query_tokens: list[str],
    query_text: str,
    index_data: dict,
) -> dict[str, float]:
    faiss_index = index_data.get("faiss_index")
    doc_ids = index_data.get("embedding_doc_ids") or []
    model_name = index_data.get("embedding_model")
    if faiss_index is None or not doc_ids or not model_name:
        return {}

    from retrieval_service.infrastructure.embedding_model import encode_text

    text = query_text.strip() or " ".join(query_tokens)
    query_vector = encode_text(text, model_name)
    scores, indices = faiss_index.search(
        np.array([query_vector], dtype=np.float32),
        len(doc_ids),
    )
    results: dict[str, float] = {}
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results[doc_ids[idx]] = float(score)
    return results


def _embedding_scores(query_tokens: list[str], index_data: dict) -> dict[str, float]:
    return score_embedding_semantic(
        query_tokens,
        index_data.get("query_text", ""),
        index_data,
    )


def fuse_serial_rerank(
    query_tokens: list[str],
    index_data: dict,
    k1: float,
    b: float,
    candidate_k: int = 100,
) -> dict[str, float]:
    bm25_scores = score_bm25(query_tokens, index_data, k1=k1, b=b)
    candidates = rank_documents(bm25_scores, candidate_k)
    candidate_ids = [doc_id for doc_id, _ in candidates]

    if index_data.get("faiss_index") is None:
        return dict(candidates)

    embedding_scores = _embedding_scores(query_tokens, index_data)
    return {doc_id: embedding_scores.get(doc_id, 0.0) for doc_id in candidate_ids}


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low, high = min(values), max(values)
    if high == low:
        return {doc_id: 1.0 for doc_id in scores}
    return {doc_id: (score - low) / (high - low) for doc_id, score in scores.items()}


def fuse_weighted(
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


def fuse_rrf(
    score_maps: list[dict[str, float]],
    k: int = 60,
) -> dict[str, float]:
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
    tfidf_scores = score_tfidf(query_tokens, index_data)
    bm25_scores = score_bm25(query_tokens, index_data, k1=k1, b=b)
    embedding_scores = _embedding_scores(query_tokens, index_data)
    return tfidf_scores, bm25_scores, embedding_scores


def run_parallel_scorers(
    query_tokens: list[str],
    index_data: dict,
    k1: float,
    b: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    def score_tfidf_task() -> dict[str, float]:
        return score_tfidf(query_tokens, index_data)

    def score_bm25_task() -> dict[str, float]:
        return score_bm25(query_tokens, index_data, k1=k1, b=b)

    def score_embedding_task() -> dict[str, float]:
        return _embedding_scores(query_tokens, index_data)

    with ThreadPoolExecutor(max_workers=3) as executor:
        tfidf_future = executor.submit(score_tfidf_task)
        bm25_future = executor.submit(score_bm25_task)
        embed_future = executor.submit(score_embedding_task)
        return tfidf_future.result(), bm25_future.result(), embed_future.result()
