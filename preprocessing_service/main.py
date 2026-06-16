"""
This is the application entry point.
It initializes the FastAPI framework, registers routers, and configures the web server.
"""
from fastapi import FastAPI
from preprocessing_service.api.routes import router

app = FastAPI(
    title="Preprocessing Service",
    description="Text cleaning, tokenization, and stemming using NLTK.",
    version="1.0.0",
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    from shared.config import SERVICE_PORTS

    uvicorn.run(
        "preprocessing_service.main:app",
        host="127.0.0.1",
        port=SERVICE_PORTS["preprocessing"],
        reload=True,
    )