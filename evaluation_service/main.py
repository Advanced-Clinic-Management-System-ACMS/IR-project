import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from evaluation_service.api.routes import router

app = FastAPI(
    title="Evaluation Service",
    description="Compute MAP, Recall, Precision@10, and nDCG.",
    version="1.0.0",
)

# Register the routes
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    from shared.config import SERVICE_PORTS

    uvicorn.run(
        "evaluation_service.main:app",
        host="127.0.0.1",
        port=SERVICE_PORTS["evaluation"],
        reload=True,
    )