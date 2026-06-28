"""
This file defines the RESTful API endpoints for the preprocessing service.
It is responsible strictly for HTTP request validation, routing, and response formatting.
"""
import os
from concurrent.futures import ProcessPoolExecutor

from fastapi import APIRouter, HTTPException
from shared.schemas import (
    DocumentInput,
    HealthResponse,
    PreprocessRequest,
    PreprocessResponse,
    ProcessedDocument,
)
from preprocessing_service.core.batch_worker import _init_worker, process_document_worker
from preprocessing_service.schemas.payloads import QueryProcessResponse
from preprocessing_service.services.document_processor import DocumentProcessorService

router = APIRouter()
processor_service = DocumentProcessorService()

PARALLEL_BATCH_THRESHOLD = 20
MAX_PREPROCESS_WORKERS = min(os.cpu_count() or 4, 8)

@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        service="preprocessing_service",
        status="ok",
        details={
            "stemmer": "PorterStemmer",
            "lemmatizer": "WordNetLemmatizer",
            "normalization_mode": processor_service.nlp_engine.normalization_mode,
            "library": "NLTK",
            "parallel_workers": MAX_PREPROCESS_WORKERS,
        },
    )

@router.post("/process", response_model=ProcessedDocument)
def process_single_document(document: DocumentInput) -> ProcessedDocument:
    """معالجة وثيقة واحدة"""
    try:
        return processor_service.process_document(document)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _process_documents_parallel(documents: list[DocumentInput]) -> list[ProcessedDocument]:
    doc_payloads = [doc.model_dump() for doc in documents]
    with ProcessPoolExecutor(
        max_workers=MAX_PREPROCESS_WORKERS,
        initializer=_init_worker,
    ) as pool:
        processed_dicts = list(pool.map(process_document_worker, doc_payloads, chunksize=32))
    return [ProcessedDocument(**item) for item in processed_dicts]


@router.post("/process-batch", response_model=PreprocessResponse)
def process_batch(request: PreprocessRequest) -> PreprocessResponse:
    """معالجة مجموعة من الوثائق (multiprocessing للدفعات الكبيرة)"""
    try:
        documents = request.documents
        if len(documents) >= PARALLEL_BATCH_THRESHOLD:
            processed = _process_documents_parallel(documents)
        else:
            processed = [processor_service.process_document(doc) for doc in documents]
        return PreprocessResponse(processed=processed, count=len(processed))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-query", response_model=QueryProcessResponse)
def process_query(query_text: str) -> QueryProcessResponse:
    """معالجة استعلام بحث (نص واحد) وإرجاع tokens بشكل محكم النوع"""
    try:
        tokens = processor_service.process_raw_query(query_text)
        return QueryProcessResponse(
            query_id="temp",
            original_text=query_text,
            tokens=tokens,
            token_count=len(tokens)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))