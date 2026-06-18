"""
This is the application entry point.
It initializes the FastAPI framework, registers routers, and configures the web server.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from indexing_service.api.routes import router

app = FastAPI(
    title="Indexing Service",
    description="Build inverted index, TF-IDF weights, BM25 stats, and optional embeddings.",
    version="1.0.0",
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    from shared.config import SERVICE_PORTS

    uvicorn.run(
        "indexing_service.main:app",
        host="127.0.0.1",
        port=SERVICE_PORTS["indexing"],
        reload=True,
    )