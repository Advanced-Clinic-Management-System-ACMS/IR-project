"""
Process-pool workers for parallel document preprocessing.
Each worker owns its own NLPEngine instance (required for multiprocessing on Windows).
"""
from __future__ import annotations

from shared.schemas import DocumentInput, ProcessedDocument

_processor = None


def _init_worker() -> None:
    global _processor
    from preprocessing_service.services.document_processor import DocumentProcessorService

    _processor = DocumentProcessorService()


def process_document_worker(doc_dict: dict) -> dict:
    if _processor is None:
        _init_worker()
    document = DocumentInput(**doc_dict)
    result: ProcessedDocument = _processor.process_document(document)
    return result.model_dump()
