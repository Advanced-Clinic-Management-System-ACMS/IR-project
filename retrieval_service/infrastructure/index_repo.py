"""
Load pre-built compressed indexes from disk at service startup / first query.
FAISS and embedding model are loaded from disk — never built on first online query.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import load_npz

from shared.config import INDEX_DIR


@dataclass
class LoadedIndex:
    dataset_name: str
    index_path: Path
    metadata: dict
    tfidf_vectorizer: object
    tfidf_matrix: object
    tfidf_doc_ids: list[str]
    bm25_model: object
    bm25_doc_ids: list[str]
    faiss_index: object | None
    embedding_doc_ids: list[str]
    embedding_model: str | None


class IndexRepository:
    _cache: dict[str, LoadedIndex] = {}

    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        self.index_dir = INDEX_DIR / dataset_name

    @staticmethod
    def _read_json_gz(path: Path):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)

    def load(self) -> LoadedIndex:
        if self.dataset_name in self._cache:
            return self._cache[self.dataset_name]

        if not self.index_dir.exists():
            raise FileNotFoundError(
                f"Index not found for dataset '{self.dataset_name}' at {self.index_dir}. "
                "Run: py scripts/run_offline_pipeline.py --ir-datasets --index-only"
            )

        metadata_path = self.index_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata.json in {self.index_dir}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("index_format") != "library_v2":
            raise FileNotFoundError(
                "Index format outdated. Rebuild with: "
                "py scripts/run_offline_pipeline.py --ir-datasets --index-only"
            )

        tfidf_vectorizer = joblib.load(self.index_dir / "tfidf_vectorizer.joblib")
        tfidf_matrix = load_npz(self.index_dir / "tfidf_matrix.npz")
        tfidf_doc_ids = self._read_json_gz(self.index_dir / "tfidf_doc_ids.json.gz")

        bm25_model = joblib.load(self.index_dir / "bm25_model.joblib")
        bm25_doc_ids = self._read_json_gz(self.index_dir / "bm25_doc_ids.json.gz")

        faiss_index = None
        embedding_doc_ids: list[str] = []
        embedding_model = metadata.get("embedding_model")
        faiss_path = self.index_dir / "faiss.index"
        doc_ids_path = self.index_dir / "embedding_doc_ids.json.gz"
        if faiss_path.exists() and doc_ids_path.exists():
            import faiss

            faiss_index = faiss.read_index(str(faiss_path))
            embedding_doc_ids = self._read_json_gz(doc_ids_path)

        loaded = LoadedIndex(
            dataset_name=self.dataset_name,
            index_path=self.index_dir,
            metadata=metadata,
            tfidf_vectorizer=tfidf_vectorizer,
            tfidf_matrix=tfidf_matrix,
            tfidf_doc_ids=tfidf_doc_ids,
            bm25_model=bm25_model,
            bm25_doc_ids=bm25_doc_ids,
            faiss_index=faiss_index,
            embedding_doc_ids=embedding_doc_ids,
            embedding_model=embedding_model,
        )
        self._cache[self.dataset_name] = loaded
        return loaded

    def as_scoring_payload(self, loaded: LoadedIndex, fusion_mode: str = "rrf") -> dict:
        return {
            "tfidf_vectorizer": loaded.tfidf_vectorizer,
            "tfidf_matrix": loaded.tfidf_matrix,
            "tfidf_doc_ids": loaded.tfidf_doc_ids,
            "bm25_model": loaded.bm25_model,
            "bm25_doc_ids": loaded.bm25_doc_ids,
            "faiss_index": loaded.faiss_index,
            "embedding_doc_ids": loaded.embedding_doc_ids,
            "embedding_type": loaded.metadata.get("embedding_type"),
            "embedding_model": loaded.embedding_model,
            "fusion_mode": fusion_mode,
            "total_docs": loaded.metadata.get("document_count", len(loaded.bm25_doc_ids)),
        }

    @classmethod
    def preload(cls, dataset_name: str) -> LoadedIndex:
        return cls(dataset_name).load()
