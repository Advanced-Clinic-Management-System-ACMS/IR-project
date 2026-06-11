"""
Offline pipeline script:
1. Load a sample or full dataset
2. Preprocess documents
3. Store raw docs in MongoDB
4. Build indexes
"""

import argparse
import json
from pathlib import Path

import requests

from shared.config import DATA_DIR, RAW_DIR, SERVICE_URLS
from shared.database import DocumentStore
from shared.schemas import DocumentInput


def load_sample_documents(limit: int = 100) -> list[DocumentInput]:
    sample_path = RAW_DIR / "sample_documents.json"
    if sample_path.exists():
        payload = json.loads(sample_path.read_text(encoding="utf-8"))
        docs = [DocumentInput(**item) for item in payload[:limit]]
        return docs

    return [
        DocumentInput(
            doc_id=f"doc_{idx}",
            text=text,
            title=f"Sample {idx}",
        )
        for idx, text in enumerate(
            [
                "Information retrieval systems help users find relevant documents.",
                "BM25 and TF-IDF are classic ranking models in search engines.",
                "Embeddings capture semantic similarity between queries and passages.",
                "Hybrid retrieval combines lexical and semantic signals.",
            ],
            start=1,
        )[:limit]
    ]


def load_from_ir_datasets(dataset_name: str, limit: int) -> list[DocumentInput]:
    import ir_datasets

    dataset = ir_datasets.load(dataset_name)
    documents: list[DocumentInput] = []

    for doc in dataset.docs_iter():
        documents.append(
            DocumentInput(
                doc_id=doc.doc_id,
                text=getattr(doc, "text", "") or "",
                title=getattr(doc, "title", None),
            )
        )
        if len(documents) >= limit:
            break

    return documents


def run_pipeline(dataset_name: str, limit: int, use_ir_datasets: bool) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if use_ir_datasets:
        documents = load_from_ir_datasets(dataset_name, limit)
    else:
        documents = load_sample_documents(limit)

    print(f"Loaded {len(documents)} documents.")

    preprocess_response = requests.post(
        f"{SERVICE_URLS['preprocessing']}/process-batch",
        json={"documents": [doc.model_dump() for doc in documents]},
        timeout=120,
    )
    preprocess_response.raise_for_status()
    processed = preprocess_response.json()["processed"]
    print(f"Preprocessed {len(processed)} documents.")

    store = DocumentStore()
    raw_payload = [
        {"doc_id": doc.doc_id, "text": doc.text, "title": doc.title}
        for doc in documents
    ]
    inserted = store.insert_documents(raw_payload)
    print(f"Stored {inserted} raw documents in MongoDB.")

    index_response = requests.post(
        f"{SERVICE_URLS['indexing']}/build-index",
        json={
            "processed_documents": processed,
            "dataset_name": dataset_name.replace("/", "_"),
            "save_embeddings": True,
        },
        timeout=300,
    )
    index_response.raise_for_status()
    print(index_response.json()["message"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the offline IR pipeline.")
    parser.add_argument("--dataset", default="default")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--ir-datasets", action="store_true")
    args = parser.parse_args()

    run_pipeline(args.dataset, args.limit, args.ir_datasets)
