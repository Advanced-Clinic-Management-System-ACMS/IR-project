"""
Offline Information Retrieval Pipeline.
Orchestrates data loading, API-driven preprocessing, bulk MongoDB storage, and API-driven indexing.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import requests
from pymongo import MongoClient

from shared.config import (
    DATA_DIR,
    RAW_DIR,
    SERVICE_URLS,
    MONGO_URI,
    MONGO_DB_NAME,
    MONGO_DOCS_COLLECTION,
)
from shared.schemas import DocumentInput

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class MongoBatchUploader:
    """Handles high-speed offline bulk inserts directly into MongoDB."""
    def __init__(self) -> None:
        self.client: MongoClient = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        self.collection = self.client[MONGO_DB_NAME][MONGO_DOCS_COLLECTION]

    def upload_documents(self, documents: list[dict[str, Any]]) -> int:
        if not documents:
            return 0
        logger.info("Clearing old documents from MongoDB...")
        self.collection.delete_many({})
        
        logger.info("Bulk inserting new documents...")
        result = self.collection.insert_many(documents)
        return len(result.inserted_ids)


def load_sample_documents(limit: int = 100) -> list[DocumentInput]:
    sample_path = RAW_DIR / "sample_documents.json"
    if sample_path.exists():
        payload = json.loads(sample_path.read_text(encoding="utf-8"))
        return [DocumentInput(**item) for item in payload[:limit]]

    # Fallback to hardcoded mock data
    mock_data = [
        "Information retrieval systems help users find relevant documents.",
        "BM25 and TF-IDF are classic ranking models in search engines.",
        "Embeddings capture semantic similarity between queries and passages.",
        "Hybrid retrieval combines lexical and semantic signals.",
    ]
    return [
        DocumentInput(doc_id=f"doc_{idx}", text=text, title=f"Sample {idx}")
        for idx, text in enumerate(mock_data, start=1)
    ][:limit]


def load_from_ir_datasets(dataset_name: str, limit: int) -> list[DocumentInput]:
    try:
        import ir_datasets
    except ImportError:
        logger.error("ir_datasets package not found. Run: pip install ir_datasets")
        sys.exit(1)

    logger.info(f"Downloading/Loading dataset '{dataset_name}' from ir_datasets...")
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

    # 1. Load Data
    logger.info("=== STEP 1: Loading Data ===")
    if use_ir_datasets:
        documents = load_from_ir_datasets(dataset_name, limit)
    else:
        documents = load_sample_documents(limit)
    logger.info(f"Loaded {len(documents)} documents into memory.")

    if not documents:
        logger.warning("No documents found. Exiting pipeline.")
        return

    # 2. Preprocessing via API
    logger.info("=== STEP 2: Preprocessing ===")
    try:
        preprocess_url = f"{SERVICE_URLS['preprocessing']}/process-batch"
        logger.info(f"Sending batch to {preprocess_url}...")
        
        preprocess_response = requests.post(
            preprocess_url,
            json={"documents": [doc.model_dump() for doc in documents]},
            timeout=120,
        )
        preprocess_response.raise_for_status()
        processed = preprocess_response.json()["processed"]
        logger.info(f"Successfully preprocessed {len(processed)} documents.")
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to Preprocessing Service. Is the server running?")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        logger.error(f"Preprocessing API returned an error: {e.response.text}")
        sys.exit(1)

    # 3. Store Raw Documents in MongoDB
    logger.info("=== STEP 3: MongoDB Storage ===")
    try:
        uploader = MongoBatchUploader()
        raw_payload = [
            {"doc_id": doc.doc_id, "text": doc.text, "title": doc.title}
            for doc in documents
        ]
        inserted = uploader.upload_documents(raw_payload)
        logger.info(f"Stored {inserted} raw documents in MongoDB.")
    except Exception as e:
        logger.error(f"MongoDB connection failed. Is MongoDB running on port 27017? Error: {e}")
        sys.exit(1)

    # 4. Build Index via API
    logger.info("=== STEP 4: Index Generation ===")
    try:
        indexing_url = f"{SERVICE_URLS['indexing']}/build-index"
        logger.info(f"Sending built request to {indexing_url}...")
        
        index_response = requests.post(
            indexing_url,
            json={
                "processed_documents": processed,
                "dataset_name": dataset_name.replace("/", "_"),
                "save_embeddings": True,
            },
            timeout=300,
        )
        index_response.raise_for_status()
        logger.info(f"Success: {index_response.json()['message']}")
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to Indexing Service. Is the server running?")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        logger.error(f"Indexing API returned an error: {e.response.text}")
        sys.exit(1)

    logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the offline IR pipeline.")
    parser.add_argument("--dataset", type=str, default="default", help="Name of the dataset")
    parser.add_argument("--limit", type=int, default=100, help="Max documents to process")
    parser.add_argument("--ir-datasets", action="store_true", help="Fetch from ir_datasets library")
    
    args = parser.parse_args()
    run_pipeline(args.dataset, args.limit, args.ir_datasets)