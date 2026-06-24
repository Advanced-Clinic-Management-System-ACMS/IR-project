"""
Offline pipeline for LoTTE.
Streams documents from ir-datasets, preprocesses in batches, saves JSONL,
and builds indexes locally (avoids HTTP payload limits on full corpus).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterator

if sys.platform == "win32" and not sys.flags.utf8_mode:
    import os

    os.execv(sys.executable, [sys.executable, "-X", "utf8", *sys.argv])

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from indexing_service.services.orchestrator import IndexingOrchestrator
from shared.config import DATA_DIR, DEFAULT_IR_DATASET, PROCESSED_DIR, RAW_DIR, SERVICE_URLS
from shared.document_db import DocumentDatabase
from shared.schemas import DocumentInput, ProcessedDocument

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def sanitize_dataset_name(dataset_name: str) -> str:
    return dataset_name.replace("/", "_")


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def append_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def iter_ir_dataset_documents(dataset_name: str, limit: int | None) -> Iterator[DocumentInput]:
    try:
        import ir_datasets
    except ImportError:
        logger.error("ir_datasets package not found. Run: pip install ir-datasets")
        sys.exit(1)

    dataset = ir_datasets.load(dataset_name)
    logger.info(
        "Dataset stats: docs=%s queries=%s qrels=%s",
        dataset.docs_count(),
        dataset.queries_count(),
        dataset.qrels_count(),
    )

    count = 0
    for doc in dataset.docs_iter():
        yield DocumentInput(
            doc_id=str(doc.doc_id),
            text=getattr(doc, "text", "") or "",
            title=getattr(doc, "title", None),
        )
        count += 1
        if limit and count >= limit:
            break


def load_sample_documents(limit: int) -> list[DocumentInput]:
    sample_path = RAW_DIR / "sample_documents.json"
    if sample_path.exists():
        payload = json.loads(sample_path.read_text(encoding="utf-8"))
        return [DocumentInput(**item) for item in payload[:limit]]

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


def preprocess_batch(documents: list[DocumentInput], batch_size: int) -> list[dict[str, Any]]:
    preprocess_url = f"{SERVICE_URLS['preprocessing']}/process-batch"
    processed: list[dict[str, Any]] = []

    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        logger.info("Preprocessing batch %s-%s of %s...", start + 1, start + len(batch), len(documents))
        response = requests.post(
            preprocess_url,
            json={"documents": [doc.model_dump() for doc in batch]},
            timeout=600,
        )
        response.raise_for_status()
        processed.extend(response.json()["processed"])

    return processed


def load_processed_from_jsonl(path: Path) -> list[ProcessedDocument]:
    documents: list[ProcessedDocument] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            documents.append(ProcessedDocument(**json.loads(line)))
    return documents


def build_index_local(dataset_name: str, processed_path: Path, save_embeddings: bool) -> dict:
    logger.info("Loading processed documents from %s ...", processed_path)
    processed = load_processed_from_jsonl(processed_path)
    logger.info("Building index for %s documents (local orchestrator)...", len(processed))
    return IndexingOrchestrator.execute_build(dataset_name, processed, save_embeddings)


def run_pipeline(
    dataset_name: str,
    limit: int | None,
    use_ir_datasets: bool,
    batch_size: int,
    save_embeddings: bool,
    resume: bool,
    index_only: bool,
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_dataset_name(dataset_name)
    raw_jsonl = RAW_DIR / f"{safe_name}.jsonl"
    processed_jsonl = PROCESSED_DIR / f"{safe_name}.jsonl"

    if index_only:
        if not processed_jsonl.exists():
            logger.error("Processed file not found: %s", processed_jsonl)
            sys.exit(1)
        logger.info("=== INDEX ONLY: skipping load/preprocess, reusing %s ===", processed_jsonl)
        try:
            metadata = build_index_local(safe_name, processed_jsonl, save_embeddings)
        except Exception as exc:
            logger.error("Indexing failed: %s", exc)
            sys.exit(1)
        logger.info("=== PIPELINE COMPLETED ===")
        logger.info("Documents indexed: %s", metadata["document_count"])
        logger.info("Indexes  : data/indexes/%s/", safe_name)
        return

    if resume:
        skipped = count_jsonl_lines(raw_jsonl)
        logger.info("Resume mode: skipping first %s documents already saved.", skipped)
    else:
        skipped = 0
        if raw_jsonl.exists():
            raw_jsonl.unlink()
        if processed_jsonl.exists():
            processed_jsonl.unlink()

    logger.info("=== STEP 1-3: Load, save raw, preprocess (streaming batches) ===")
    if use_ir_datasets:
        doc_iter = iter_ir_dataset_documents(dataset_name, limit)
        batch: list[DocumentInput] = []
        total_saved = skipped
        seen = 0

        for document in doc_iter:
            seen += 1
            if seen <= skipped:
                continue
            batch.append(document)
            if len(batch) < batch_size:
                continue

            try:
                processed = preprocess_batch(batch, batch_size)
            except requests.exceptions.ConnectionError:
                logger.error("Preprocessing service is offline. Start: py preprocessing_service\\main.py")
                sys.exit(1)
            except requests.exceptions.HTTPError as exc:
                logger.error("Preprocessing failed: %s", exc.response.text)
                sys.exit(1)

            append_jsonl(raw_jsonl, [doc.model_dump() for doc in batch])
            append_jsonl(processed_jsonl, processed)
            DocumentDatabase.upsert_documents(safe_name, batch)
            total_saved += len(batch)
            logger.info("Saved %s documents so far.", total_saved)
            batch = []

        if batch:
            try:
                processed = preprocess_batch(batch, batch_size)
            except requests.exceptions.ConnectionError:
                logger.error("Preprocessing service is offline. Start: py preprocessing_service\\main.py")
                sys.exit(1)
            append_jsonl(raw_jsonl, [doc.model_dump() for doc in batch])
            append_jsonl(processed_jsonl, processed)
            DocumentDatabase.upsert_documents(safe_name, batch)
            total_saved += len(batch)
            logger.info("Saved %s documents so far.", total_saved)
    else:
        documents = load_sample_documents(limit or 100)
        processed = preprocess_batch(documents, batch_size)
        append_jsonl(raw_jsonl, [doc.model_dump() for doc in documents])
        append_jsonl(processed_jsonl, processed)
        DocumentDatabase.upsert_documents(safe_name, documents)
        total_saved = len(documents)

    if total_saved == 0:
        logger.warning("No documents processed. Exiting.")
        return

    logger.info("=== STEP 4: Index generation (full corpus) ===")
    try:
        metadata = build_index_local(safe_name, processed_jsonl, save_embeddings)
    except Exception as exc:
        logger.error("Indexing failed: %s", exc)
        sys.exit(1)

    logger.info("=== PIPELINE COMPLETED ===")
    logger.info("Documents indexed: %s", metadata["document_count"])
    logger.info("Raw docs : %s", raw_jsonl)
    logger.info("Processed: %s", processed_jsonl)
    logger.info("Indexes  : data/indexes/%s/", safe_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the offline IR pipeline for LoTTE.")
    parser.add_argument("--dataset", type=str, default=DEFAULT_IR_DATASET, help="ir-datasets identifier")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional document limit for quick tests (omit for full LoTTE corpus)",
    )
    parser.add_argument("--ir-datasets", action="store_true", help="Load documents from ir-datasets")
    parser.add_argument("--batch-size", type=int, default=500, help="Preprocessing batch size")
    parser.add_argument("--no-embeddings", action="store_true", help="Skip embedding index generation")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip documents already saved in JSONL (use after interrupted runs)",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Rebuild index from existing processed JSONL (skip load/preprocess)",
    )
    args = parser.parse_args()

    run_pipeline(
        dataset_name=args.dataset,
        limit=args.limit,
        use_ir_datasets=args.ir_datasets,
        batch_size=args.batch_size,
        save_embeddings=not args.no_embeddings,
        resume=args.resume,
        index_only=args.index_only,
    )
