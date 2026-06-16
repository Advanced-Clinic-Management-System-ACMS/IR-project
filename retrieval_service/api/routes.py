"""
This file defines the RESTful API endpoints for the retrieval service.
It strictly handles HTTP requests and strongly-typed responses.
"""
from fastapi import APIRouter, HTTPException
from shared.schemas import SearchRequest, SearchResponse, HealthResponse
from retrieval_service.services.engine import RetrievalEngineService

router = APIRouter()

# Instantiate the engine (loads index into memory once)
try:
    engine = RetrievalEngineService("default")
except FileNotFoundError:
    engine = None

@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    status = "ok" if engine else "index_missing"
    return HealthResponse(service="retrieval_service", status=status)

@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    if not engine:
        raise HTTPException(status_code=500, detail="Index not found. Please run indexing_service first.")
    
    try:
        results, elapsed_ms = engine.execute_search(request)
        return SearchResponse(
            query=request.query,
            model=request.model,
            results=results,
            elapsed_ms=elapsed_ms
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))