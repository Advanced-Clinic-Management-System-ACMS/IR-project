"""
Compressed on-disk index artifacts (examiner requirement: no build on first query).
"""
from __future__ import annotations

import gzip
import json

import joblib
import numpy as np
from scipy.sparse import load_npz, save_npz

from shared.config import INDEX_DIR, PROCESSED_DIR


class IndexStorageAdapter:
    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        self.output_dir = INDEX_DIR / dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_json_gz(self, filename: str, payload: object) -> None:
        path = self.output_dir / filename
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)

    def save_joblib(self, filename: str, payload: object) -> None:
        joblib.dump(payload, self.output_dir / filename, compress=("gzip", 3))

    def save_sparse_matrix(self, filename: str, matrix) -> None:
        save_npz(self.output_dir / filename, matrix, compressed=True)

    def save_embeddings_compressed(self, filename: str, matrix: np.ndarray) -> None:
        np.savez_compressed(self.output_dir / filename, embeddings=matrix)

    def save_faiss_index(self, index, filename: str = "faiss.index") -> None:
        import faiss

        faiss.write_index(index, str(self.output_dir / filename))

    def save_json(self, filename: str, payload: object) -> None:
        path = self.output_dir / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_processed_documents(self, documents: list[dict]) -> None:
        processed_path = PROCESSED_DIR / f"{self.dataset_name}.json"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        processed_path.write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
