"""
This is the application entry point.
It initializes the FastAPI framework and mounts the UI web routes.
"""
from fastapi import FastAPI
from ui_gateway.api.routes import router

app = FastAPI(
    title="IR System UI Gateway",
    description="Frontend Gateway for the Information Retrieval System.",
    version="1.0.0",
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    from shared.config import SERVICE_PORTS

    uvicorn.run(
        "ui_gateway.main:app",
        host="127.0.0.1",
        port=SERVICE_PORTS.get("ui_gateway", 8000), 
        reload=True,
    )