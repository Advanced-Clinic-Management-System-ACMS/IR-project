"""
Infrastructure adapter for reading pre-built indexes from disk.
This is the only place in retrieval_service that touches index files.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared.config import INDEX_DIR


@dataclass
class LoadedIndex:
    dataset_name: str
    index_path: Path
    inverted_index: dict[str, dict[str, int]]
    vocabulary: list[str]
    doc_lengths: dict[str, int]
    idf: dict[str, float]
    tf_idf_vectors: dict[str, dict[str, float]]
    avg_doc_length: float
    total_docs: int
    embeddings: np.ndarray | None
    embedding_doc_ids: list[str]


class IndexRepository:
    _cache: dict[str, LoadedIndex] = {}

    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        self.index_dir = INDEX_DIR / dataset_name

    def _read_json(self, filename: str):
        path = self.index_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing index file: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def load(self) -> LoadedIndex:
        if self.dataset_name in self._cache:
            return self._cache[self.dataset_name]

        if not self.index_dir.exists():
            raise FileNotFoundError(
                f"Index not found for dataset '{self.dataset_name}' at {self.index_dir}. "
                "Run scripts/run_offline_pipeline.py first."
            )

        metadata = self._read_json("metadata.json")
        bm25_stats = self._read_json("bm25_stats.json")
        embeddings = None
        embedding_doc_ids: list[str] = []
        embeddings_path = self.index_dir / "embeddings.npy"
        if embeddings_path.exists():
            embeddings = np.load(embeddings_path)
            embedding_doc_ids = self._read_json("embedding_doc_ids.json")

        loaded = LoadedIndex(
            dataset_name=self.dataset_name,
            index_path=self.index_dir,
            inverted_index=self._read_json("inverted_index.json"),
            vocabulary=self._read_json("vocabulary.json"),
            doc_lengths=self._read_json("doc_lengths.json"),
            idf=self._read_json("idf.json"),
            tf_idf_vectors=self._read_json("tf_idf_vectors.json"),
            avg_doc_length=float(bm25_stats.get("avg_doc_length", metadata.get("avg_doc_length", 1.0))),
            total_docs=int(metadata.get("document_count", len(self._read_json("doc_lengths.json")))),
            embeddings=embeddings,
            embedding_doc_ids=embedding_doc_ids,
        )
        self._cache[self.dataset_name] = loaded
        return loaded

    def as_scoring_payload(self, loaded: LoadedIndex) -> dict:
        metadata_path = self.index_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        return {
            "inverted_index": loaded.inverted_index,
            "vocabulary": loaded.vocabulary,
            "doc_lengths": loaded.doc_lengths,
            "idf": loaded.idf,
            "tf_idf_vectors": loaded.tf_idf_vectors,
            "avg_doc_length": loaded.avg_doc_length,
            "embeddings": loaded.embeddings,
            "embedding_doc_ids": loaded.embedding_doc_ids,
            "embedding_type": metadata.get("embedding_type", "vocabulary_tfidf"),
            "embedding_model": metadata.get("embedding_model"),
        }
