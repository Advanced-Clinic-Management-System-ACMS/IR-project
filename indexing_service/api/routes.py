"""
This file defines the RESTful API endpoints for the indexing service.
It is responsible strictly for HTTP request validation, routing, and response formatting.
"""
from fastapi import APIRouter, HTTPException
from shared.schemas import BuildIndexRequest, BuildIndexResponse, HealthResponse
from indexing_service.services.orchestrator import IndexingOrchestrator

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(service="indexing_service", status="ok")

@router.post("/build-index", response_model=BuildIndexResponse)
def build_index(request: BuildIndexRequest) -> BuildIndexResponse:
    if not request.processed_documents:
        raise HTTPException(status_code=400, detail="processed_documents cannot be empty.")

    try:
        metadata = IndexingOrchestrator.execute_build(
            dataset_name=request.dataset_name,
            documents=request.processed_documents,
            save_embeddings=request.save_embeddings
        )
        
        return BuildIndexResponse(
            dataset_name=metadata["dataset_name"],
            document_count=metadata["document_count"],
            vocabulary_size=metadata["vocabulary_size"],
            index_path=metadata["index_path"],
            message="Index built successfully.",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))