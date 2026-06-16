"""
This file defines the RESTful API endpoints for the query refinement service.
It strictly handles HTTP requests, payload validation, and response formatting.
"""
from fastapi import APIRouter
from shared.schemas import HealthResponse
from query_refinement.schemas.payloads import RefineRequest, RefineResponse
from query_refinement.services.refinement_service import RefinementOrchestrator

router = APIRouter()
orchestrator = RefinementOrchestrator()

@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(service="query_refinement_service", status="ok")

@router.post("/refine", response_model=RefineResponse)
def refine_query(request: RefineRequest) -> RefineResponse:
    return orchestrator.execute_refinement(request)