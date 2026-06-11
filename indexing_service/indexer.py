import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from shared.config import INDEX_DIR, PROCESSED_DIR
from shared.schemas import ProcessedDocument


class InvertedIndexBuilder:
    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        self.output_dir = INDEX_DIR / dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(self, documents: list[ProcessedDocument], save_embeddings: bool = False) -> dict:
        if not documents:
            raise ValueError("No documents provided for indexing.")

        doc_freq: dict[str, int] = defaultdict(int)
        inverted_index: dict[str, dict[str, int]] = defaultdict(dict)
        doc_lengths: dict[str, int] = {}
        processed_payload: list[dict] = []

        for doc in documents:
            term_counts = Counter(doc.tokens)
            doc_lengths[doc.doc_id] = len(doc.tokens)
            processed_payload.append(doc.model_dump())

            for term, tf in term_counts.items():
                inverted_index[term][doc.doc_id] = tf
                doc_freq[term] += 1

        total_docs = len(documents)
        avg_doc_length = sum(doc_lengths.values()) / total_docs
        vocabulary = sorted(inverted_index.keys())

        idf = {
            term: math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in doc_freq.items()
        }

        tf_idf_vectors = self._build_tf_idf_vectors(documents, idf, vocabulary)
        bm25_stats = {
            "doc_lengths": doc_lengths,
            "avg_doc_length": avg_doc_length,
            "doc_freq": dict(doc_freq),
            "idf": idf,
        }

        self._save_json("inverted_index.json", {k: dict(v) for k, v in inverted_index.items()})
        self._save_json("vocabulary.json", vocabulary)
        self._save_json("doc_lengths.json", doc_lengths)
        self._save_json("idf.json", idf)
        self._save_json("bm25_stats.json", bm25_stats)
        self._save_json("tf_idf_vectors.json", tf_idf_vectors)

        if save_embeddings:
            embeddings = self._build_simple_embeddings(documents, vocabulary)
            np.save(self.output_dir / "embeddings.npy", embeddings)
            self._save_json("embedding_doc_ids.json", [doc.doc_id for doc in documents])

        processed_path = PROCESSED_DIR / f"{self.dataset_name}.json"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        processed_path.write_text(json.dumps(processed_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        metadata = {
            "dataset_name": self.dataset_name,
            "document_count": total_docs,
            "vocabulary_size": len(vocabulary),
            "avg_doc_length": avg_doc_length,
            "index_path": str(self.output_dir),
        }
        self._save_json("metadata.json", metadata)
        return metadata

    def _build_tf_idf_vectors(
        self,
        documents: list[ProcessedDocument],
        idf: dict[str, float],
        vocabulary: list[str],
    ) -> dict[str, dict[str, float]]:
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

    def _build_simple_embeddings(
        self,
        documents: list[ProcessedDocument],
        vocabulary: list[str],
    ) -> np.ndarray:
        """Lightweight bag-of-words style vectors until a real embedding model is added."""
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

    def _save_json(self, filename: str, payload: object) -> None:
        path = self.output_dir / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
