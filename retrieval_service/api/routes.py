"""
REST API controller for the retrieval microservice.
"""
from fastapi import APIRouter, HTTPException

from shared.schemas import HealthResponse, SearchRequest, SearchResponse
from retrieval_service.services.engine import RetrievalEngine

router = APIRouter()
engine = RetrievalEngine()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        service="retrieval_service",
        status="ok",
        details={"architecture": "index_repo + document_repo + preprocess_client"},
    )


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    try:
        return engine.search(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
