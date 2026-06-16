# Pydantic models for strict type validation
from pydantic import BaseModel, Field

class EvaluateRequest(BaseModel):
    runs: dict[str, dict[str, list[str]]] = Field(..., description="Model predictions mapped to queries and retrieved documents")
    qrels: dict[str, dict[str, int]] = Field(..., description="Ground truth relevance scores")
    k: int = Field(default=10, ge=1, description="Cutoff rank for evaluation metrics")

class ModelMetrics(BaseModel):
    MAP: float
    Recall: float
    PrecisionAt10: float
    nDCGAt10: float

class EvaluateResponse(BaseModel):
    k: int
    metrics: dict[str, ModelMetrics]