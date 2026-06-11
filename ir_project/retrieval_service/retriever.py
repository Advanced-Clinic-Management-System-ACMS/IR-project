import json
import math
import time
from pathlib import Path

import numpy as np

from preprocessing_service.processor import TextPreprocessor
from shared.config import DEFAULT_BM25_B, DEFAULT_BM25_K1, INDEX_DIR


class IndexRepository:
    def __init__(self, dataset_name: str = "default") -> None:
        self.dataset_name = dataset_name
        self.base_path = INDEX_DIR / dataset_name
        self._load()

    def _load_json(self, filename: str):
        path = self.base_path / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing index file: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _load(self) -> None:
        self.inverted_index = self._load_json("inverted_index.json")
        self.tf_idf_vectors = self._load_json("tf_idf_vectors.json")
        self.bm25_stats = self._load_json("bm25_stats.json")
        self.doc_lengths = self._load_json("doc_lengths.json")

        embedding_path = self.base_path / "embeddings.npy"
        if embedding_path.exists():
            self.embeddings = np.load(embedding_path)
            self.embedding_doc_ids = self._load_json("embedding_doc_ids.json")
        else:
            self.embeddings = None
            self.embedding_doc_ids = []


class RetrievalEngine:
    def __init__(self, dataset_name: str = "default") -> None:
        self.repository = IndexRepository(dataset_name)
        self.preprocessor = TextPreprocessor()

    def search(
        self,
        query: str,
        model: str,
        top_k: int = 10,
        bm25_k1: float | None = None,
        bm25_b: float | None = None,
        hybrid_weights: dict[str, float] | None = None,
    ) -> tuple[list[tuple[str, float]], float]:
        start = time.perf_counter()
        query_tokens = self.preprocessor.process_text(query)

        if model == "tf_idf":
            scores = self._search_tf_idf(query_tokens)
        elif model == "bm25":
            scores = self._search_bm25(
                query_tokens,
                k1=bm25_k1 or DEFAULT_BM25_K1,
                b=bm25_b or DEFAULT_BM25_B,
            )
        elif model == "embedding":
            scores = self._search_embedding(query_tokens)
        elif model in {"hybrid_serial", "hybrid_parallel"}:
            scores = self._search_hybrid(
                query_tokens,
                parallel=model == "hybrid_parallel",
                weights=hybrid_weights,
                bm25_k1=bm25_k1 or DEFAULT_BM25_K1,
                bm25_b=bm25_b or DEFAULT_BM25_B,
            )
        else:
            raise ValueError(f"Unsupported model: {model}")

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        elapsed_ms = (time.perf_counter() - start) * 1000
        return ranked, elapsed_ms

    def _search_tf_idf(self, query_tokens: list[str]) -> dict[str, float]:
        query_counts: dict[str, int] = {}
        for token in query_tokens:
            query_counts[token] = query_counts.get(token, 0) + 1

        total_query_terms = len(query_tokens) or 1
        idf_map = self.repository.bm25_stats["idf"]
        query_vector = {
            term: (count / total_query_terms) * idf_map.get(term, 0.0)
            for term, count in query_counts.items()
        }

        scores: dict[str, float] = {}
        candidate_docs = set()
        for term in query_vector:
            candidate_docs.update(self.repository.inverted_index.get(term, {}).keys())

        for doc_id in candidate_docs:
            doc_vector = self.repository.tf_idf_vectors.get(doc_id, {})
            scores[doc_id] = self._cosine_similarity(query_vector, doc_vector)

        return scores

    def _search_bm25(
        self,
        query_tokens: list[str],
        k1: float,
        b: float,
    ) -> dict[str, float]:
        stats = self.repository.bm25_stats
        idf_map = stats["idf"]
        avg_dl = stats["avg_doc_length"]
        scores: dict[str, float] = {}

        for term in set(query_tokens):
            if term not in self.repository.inverted_index:
                continue

            idf = idf_map.get(term, 0.0)
            postings = self.repository.inverted_index[term]

            for doc_id, tf in postings.items():
                doc_len = self.repository.doc_lengths.get(doc_id, 0)
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / avg_dl))
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (numerator / denominator)

        return scores

    def _search_embedding(self, query_tokens: list[str]) -> dict[str, float]:
        if self.repository.embeddings is None:
            raise ValueError("Embeddings were not built for this dataset.")

        vocabulary = list(self.repository.bm25_stats["doc_freq"].keys())
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

        similarities = self.repository.embeddings @ query_vec
        return {
            doc_id: float(score)
            for doc_id, score in zip(self.repository.embedding_doc_ids, similarities, strict=True)
        }

    def _search_hybrid(
        self,
        query_tokens: list[str],
        parallel: bool,
        weights: dict[str, float] | None,
        bm25_k1: float,
        bm25_b: float,
    ) -> dict[str, float]:
        default_weights = {"tf_idf": 0.34, "bm25": 0.33, "embedding": 0.33}
        weights = weights or default_weights

        tf_idf_scores = self._search_tf_idf(query_tokens)
        bm25_scores = self._search_bm25(query_tokens, k1=bm25_k1, b=bm25_b)

        if self.repository.embeddings is not None:
            embedding_scores = self._search_embedding(query_tokens)
        else:
            embedding_scores = {}

        if parallel:
            combined = self._fuse_parallel(
                [tf_idf_scores, bm25_scores, embedding_scores],
                list(weights.values()),
            )
        else:
            combined = self._fuse_serial(
                tf_idf_scores,
                bm25_scores,
                embedding_scores,
                weights,
            )

        return combined

    @staticmethod
    def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
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

    def _fuse_serial(
        self,
        tf_idf_scores: dict[str, float],
        bm25_scores: dict[str, float],
        embedding_scores: dict[str, float],
        weights: dict[str, float],
    ) -> dict[str, float]:
        first = self._normalize_scores(tf_idf_scores)
        second = self._normalize_scores(bm25_scores)
        third = self._normalize_scores(embedding_scores)

        all_docs = set(first) | set(second) | set(third)
        combined: dict[str, float] = {}

        for doc_id in all_docs:
            combined[doc_id] = (
                weights.get("tf_idf", 0.34) * first.get(doc_id, 0.0)
                + weights.get("bm25", 0.33) * second.get(doc_id, 0.0)
                + weights.get("embedding", 0.33) * third.get(doc_id, 0.0)
            )

        return combined

    def _fuse_parallel(
        self,
        score_lists: list[dict[str, float]],
        weights: list[float],
    ) -> dict[str, float]:
        normalized = [self._normalize_scores(scores) for scores in score_lists]
        all_docs = set().union(*normalized)
        combined: dict[str, float] = {}

        for doc_id in all_docs:
            combined[doc_id] = sum(
                weight * score_map.get(doc_id, 0.0)
                for weight, score_map in zip(weights, normalized, strict=False)
            )

        return combined

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
