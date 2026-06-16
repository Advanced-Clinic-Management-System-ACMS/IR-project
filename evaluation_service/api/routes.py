# (Controller) Handles HTTP requests and responses
from fastapi import APIRouter
from shared.schemas import HealthResponse
from evaluation_service.schemas.payloads import EvaluateRequest, EvaluateResponse
from evaluation_service.services.evaluator import EvaluationService

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(service="evaluation_service", status="ok")

@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(payload: EvaluateRequest) -> EvaluateResponse:
    # Notice how clean this is. The controller does zero math.
    report = EvaluationService.generate_report(payload)
    return EvaluateResponse(k=payload.k, metrics=report)