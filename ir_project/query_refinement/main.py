from fastapi import FastAPI

from shared.schemas import HealthResponse

app = FastAPI(
    title="Query Refinement Service",
    description="Query reformulation, spelling correction, and suggestions.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(service="query_refinement_service", status="ok")


@app.post("/refine")
def refine_query(payload: dict) -> dict:
    query = payload.get("query", "").strip()
    history = payload.get("history", [])

    refined = query
    suggestions = []

    if history:
        suggestions.append(history[-1])

    return {
        "original_query": query,
        "refined_query": refined,
        "suggestions": suggestions,
        "notes": "Add spell correction and synonym expansion in the next iteration.",
    }


if __name__ == "__main__":
    import uvicorn

    from shared.config import SERVICE_PORTS

    uvicorn.run(
        "query_refinement.main:app",
        host="127.0.0.1",
        port=SERVICE_PORTS["query_refinement"],
        reload=True,
    )
