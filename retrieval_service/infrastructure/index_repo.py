"""
This file manages reading the inverted index JSON and Numpy arrays from the file system.
"""
import json
from pathlib import Path
import numpy as np
from shared.config import INDEX_DIR

class IndexRepository:
    def __init__(self, dataset_name: str = "default") -> None:
        self.base_path = INDEX_DIR / dataset_name

    def load_index_data(self) -> dict:
        def _load_json(filename: str):
            path = self.base_path / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing index file: {path}")
            return json.loads(path.read_text(encoding="utf-8"))

        data = {
            "inverted_index": _load_json("inverted_index.json"),
            "tf_idf_vectors": _load_json("tf_idf_vectors.json"),
            "bm25_stats": _load_json("bm25_stats.json"),
            "doc_lengths": _load_json("doc_lengths.json")
        }

        embedding_path = self.base_path / "embeddings.npy"
        if embedding_path.exists():
            data["embeddings"] = np.load(embedding_path)
            data["embedding_doc_ids"] = _load_json("embedding_doc_ids.json")
        else:
            data["embeddings"] = None
            data["embedding_doc_ids"] = []
            
        return data