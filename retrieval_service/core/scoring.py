"""
This file contains the core enterprise business rules for scoring documents.
It implements TF-IDF, BM25, Embedding similarity, and Hybrid fusion math.
It is entirely isolated from any infrastructure or application dependencies.
"""
import math
import numpy as np

class ScoringCore:
    @staticmethod
    def compute_tf_idf(query_tokens: list[str], inverted_index: dict, tf_idf_vectors: dict, idf_map: dict) -> dict[str, float]:
        query_counts: dict[str, int] = {}
        for token in query_tokens:
            query_counts[token] = query_counts.get(token, 0) + 1

        total_query_terms = len(query_tokens) or 1
        query_vector = {
            term: (count / total_query_terms) * idf_map.get(term, 0.0)
            for term, count in query_counts.items()
        }

        scores: dict[str, float] = {}
        candidate_docs = set()
        for term in query_vector:
            candidate_docs.update(inverted_index.get(term, {}).keys())

        for doc_id in candidate_docs:
            doc_vector = tf_idf_vectors.get(doc_id, {})
            scores[doc_id] = ScoringCore._cosine_similarity(query_vector, doc_vector)

        return scores

    @staticmethod
    def compute_bm25(query_tokens: list[str], inverted_index: dict, bm25_stats: dict, doc_lengths: dict, k1: float, b: float) -> dict[str, float]:
        idf_map = bm25_stats["idf"]
        avg_dl = bm25_stats["avg_doc_length"]
        scores: dict[str, float] = {}

        for term in set(query_tokens):
            if term not in inverted_index:
                continue

            idf = idf_map.get(term, 0.0)
            postings = inverted_index[term]

            for doc_id, tf in postings.items():
                doc_len = doc_lengths.get(doc_id, 0)
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / avg_dl))
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (numerator / denominator)

        return scores

    @staticmethod
    def compute_embedding(query_tokens: list[str], embeddings: np.ndarray, embedding_doc_ids: list, vocabulary: list[str]) -> dict[str, float]:
        term_to_index = {term: idx for idx, term in enumerate(vocabulary)}
        query_vec = np.zeros(len(vocabulary), dtype=np.float32)

        counts: dict[str, int] = {}
        for token in query_tokens:
            counts[token] = counts.get(token, 0) + 1

        total = len(query_tokens) or 1
        for term, tf in counts.items():
            idx = term_to_index.get(term)
            if idx is not None:
                query_vec[idx] = tf / total

        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec /= norm

        similarities = embeddings @ query_vec
        return {
            doc_id: float(score)
            for doc_id, score in zip(embedding_doc_ids, similarities, strict=True)
        }

    @staticmethod
    def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        max_score = max(scores.values())
        min_score = min(scores.values())
        if math.isclose(max_score, min_score):
            return {doc_id: 1.0 for doc_id in scores}
        return {
            doc_id: (score - min_score) / (max_score - min_score)
            for doc_id, score in scores.items()
        }

    @staticmethod
    def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
        common_terms = set(vec_a) & set(vec_b)
        if not common_terms:
            return 0.0
        dot = sum(vec_a[term] * vec_b[term] for term in common_terms)
        norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
        norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)