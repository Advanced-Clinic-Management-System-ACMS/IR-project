"""
This file defines the RESTful API endpoints for the preprocessing service.
It is responsible strictly for HTTP request validation, routing, and response formatting.
"""
from fastapi import APIRouter, HTTPException
from shared.schemas import (
    DocumentInput,
    HealthResponse,
    PreprocessRequest,
    PreprocessResponse,
    ProcessedDocument,
)
from preprocessing_service.schemas.payloads import QueryProcessResponse
from preprocessing_service.services.document_processor import DocumentProcessorService

router = APIRouter()
processor_service = DocumentProcessorService()

@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        service="preprocessing_service",
        status="ok",
        details={"stemmer": "PorterStemmer", "library": "NLTK"},
    )

@router.post("/process", response_model=ProcessedDocument)
def process_single_document(document: DocumentInput) -> ProcessedDocument:
    """معالجة وثيقة واحدة"""
    try:
        return processor_service.process_document(document)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-batch", response_model=PreprocessResponse)
def process_batch(request: PreprocessRequest) -> PreprocessResponse:
    """معالجة مجموعة من الوثائق"""
    try:
        processed = [processor_service.process_document(doc) for doc in request.documents]
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