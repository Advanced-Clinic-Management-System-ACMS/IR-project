"""
This is the application entry point.
It initializes the FastAPI framework, registers routers, and configures the web server.
"""
from fastapi import FastAPI
from query_refinement.api.routes import router

app = FastAPI(
    title="Query Refinement Service",
    description="Query reformulation, spelling correction, and synonym expansion.",
    version="1.0.0",
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    from shared.config import SERVICE_PORTS

    uvicorn.run(
        "query_refinement.main:app",
        host="127.0.0.1",
        port=SERVICE_PORTS["query_refinement"],
        reload=True,
    )