"""
This is the application entry point.
It initializes the FastAPI framework, registers routers, and configures the web server.
"""
from fastapi import FastAPI
from retrieval_service.api.routes import router

app = FastAPI(
    title="Retrieval Service",
    description="Core Search Engine orchestrating TF-IDF, BM25, and Embeddings.",
    version="1.0.0",
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    from shared.config import SERVICE_PORTS
    
    uvicorn.run(
        "retrieval_service.main:app",
        host="127.0.0.1",
        port=SERVICE_PORTS["retrieval"],
        reload=True,
    )