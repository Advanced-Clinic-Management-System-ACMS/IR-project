import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI

from retrieval_service.api.routes import router

app = FastAPI(
    title="IR Retrieval Service",
    description="Search over pre-built indexes with local document files.",
    version="2.0.0",
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
