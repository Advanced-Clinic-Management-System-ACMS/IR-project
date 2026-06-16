"""
This file handles all infrastructure-level disk operations.
It is responsible for writing JSON payloads and NumPy arrays to the file system.
The core business logic should never interact with this directly.
"""
import json
from pathlib import Path
import numpy as np
from shared.config import INDEX_DIR, PROCESSED_DIR

class IndexStorageAdapter:
    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
        self.output_dir = INDEX_DIR / dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, filename: str, payload: object) -> None:
        path = self.output_dir / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_numpy(self, filename: str, matrix: np.ndarray) -> None:
        np.save(self.output_dir / filename, matrix)

    def save_processed_documents(self, documents: list[dict]) -> None:
        processed_path = PROCESSED_DIR / f"{self.dataset_name}.json"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        processed_path.write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")