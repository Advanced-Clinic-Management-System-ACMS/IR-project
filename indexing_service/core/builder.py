"""
This file contains the core enterprise logic for Information Retrieval indexing.
It computes inverted indexes, TF-IDF weights, BM25 statistics, and embeddings.
It is completely isolated from external dependencies like databases or file systems.
"""
import math
from collections import Counter, defaultdict
import numpy as np
from shared.schemas import ProcessedDocument

class IndexBuilderCore:
    @staticmethod
    def build_core_index(documents: list[ProcessedDocument]) -> dict:
        doc_freq: dict[str, int] = defaultdict(int)
        inverted_index: dict[str, dict[str, int]] = defaultdict(dict)
        doc_lengths: dict[str, int] = {}
        
        for doc in documents:
            term_counts = Counter(doc.tokens)
            doc_lengths[doc.doc_id] = len(doc.tokens)
            for term, tf in term_counts.items():
                inverted_index[term][doc.doc_id] = tf
                doc_freq[term] += 1

        total_docs = len(documents)
        avg_doc_length = sum(doc_lengths.values()) / (total_docs or 1)
        vocabulary = sorted(inverted_index.keys())

        idf = {
            term: math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in doc_freq.items()
        }

        bm25_stats = {
            "doc_lengths": doc_lengths,
            "avg_doc_length": avg_doc_length,
            "doc_freq": dict(doc_freq),
            "idf": idf,
        }

        return {
            "inverted_index": {k: dict(v) for k, v in inverted_index.items()},
            "vocabulary": vocabulary,
            "doc_lengths": doc_lengths,
            "idf": idf,
            "bm25_stats": bm25_stats,
            "total_docs": total_docs,
            "avg_doc_length": avg_doc_length
        }

    @staticmethod
    def build_tf_idf_vectors(documents: list[ProcessedDocument], idf: dict[str, float], vocabulary: list[str]) -> dict[str, dict[str, float]]:
        term_to_index = {term: idx for idx, term in enumerate(vocabulary)}
        vectors: dict[str, dict[str, float]] = {}

        for doc in documents:
            counts = Counter(doc.tokens)
            total_terms = len(doc.tokens) or 1
            sparse_vector: dict[str, float] = {}

            for term, tf in counts.items():
                if term not in term_to_index:
                    continue
                tf_weight = tf / total_terms
                sparse_vector[term] = tf_weight * idf.get(term, 0.0)

            vectors[doc.doc_id] = sparse_vector
        return vectors

    @staticmethod
    def build_simple_embeddings(documents: list[ProcessedDocument], vocabulary: list[str]) -> np.ndarray:
        term_to_index = {term: idx for idx, term in enumerate(vocabulary)}
        matrix = np.zeros((len(documents), len(vocabulary)), dtype=np.float32)

        for row, doc in enumerate(documents):
            counts = Counter(doc.tokens)
            total = len(doc.tokens) or 1
            for term, tf in counts.items():
                col = term_to_index.get(term)
                if col is not None:
                    matrix[row, col] = tf / total

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms