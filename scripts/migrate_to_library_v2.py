"""
Migrate existing LoTTE processed JSONL + embeddings to library_v2 index format.
Reuses embeddings.npy when available (no re-encoding).
Also populates SQLite documents.db from processed JSONL.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from indexing_service.services.orchestrator import IndexingOrchestrator
from shared.config import DEFAULT_DATASET_NAME, INDEX_DIR, PROCESSED_DIR
from shared.document_db import DocumentDatabase
from shared.schemas import ProcessedDocument


def load_processed_jsonl(path: Path) -> list[ProcessedDocument]:
    documents: list[ProcessedDocument] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            documents.append(ProcessedDocument(**json.loads(line)))
    return documents


def convert_legacy_embeddings(dataset_name: str) -> None:
    index_dir = INDEX_DIR / dataset_name
    legacy_npy = index_dir / "embeddings.npy"
    legacy_ids = index_dir / "embedding_doc_ids.json"
    if not legacy_npy.exists():
        return
    embeddings = np.load(legacy_npy)
    if legacy_ids.exists():
        doc_ids = json.loads(legacy_ids.read_text(encoding="utf-8"))
    else:
        doc_ids = []
    np.savez_compressed(index_dir / "embeddings.npz", embeddings=embeddings)
    import gzip

    with gzip.open(index_dir / "embedding_doc_ids.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(doc_ids, handle)
    from indexing_service.core.builder import IndexBuilderCore
    from indexing_service.infrastructure.storage import IndexStorageAdapter

    storage = IndexStorageAdapter(dataset_name)
    faiss_index = IndexBuilderCore.build_faiss_index(embeddings)
    storage.save_faiss_index(faiss_index)
    print(f"Converted legacy embeddings + built faiss.index ({len(doc_ids)} docs)")


def main() -> None:
    dataset_name = DEFAULT_DATASET_NAME
    processed_path = PROCESSED_DIR / f"{dataset_name}.jsonl"
    if not processed_path.exists():
        print(f"Missing {processed_path}")
        sys.exit(1)

    print("Loading processed documents...")
    documents = load_processed_jsonl(processed_path)
    print(f"Loaded {len(documents)} documents")

    print("Populating SQLite documents.db...")
    count = DocumentDatabase.upsert_from_processed(
        dataset_name,
        [doc.model_dump() for doc in documents],
    )
    print(f"SQLite documents: {count}")

    convert_legacy_embeddings(dataset_name)

    print("Building library_v2 indexes (sklearn + rank_bm25)...")
    metadata = IndexingOrchestrator.execute_build(dataset_name, documents, save_embeddings=True)
    print("Done:", metadata)


if __name__ == "__main__":
    main()
