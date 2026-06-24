import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI

from retrieval_service.api.routes import router
from shared.config import DEFAULT_DATASET_NAME, DEFAULT_EMBEDDING_MODEL


@asynccontextmanager
async def lifespan(app: FastAPI):
    from retrieval_service.infrastructure.embedding_model import preload_model
    from retrieval_service.infrastructure.index_repo import IndexRepository

    try:
        loaded = IndexRepository.preload(DEFAULT_DATASET_NAME)
        if loaded.embedding_model:
            preload_model(loaded.embedding_model)
        else:
            preload_model(DEFAULT_EMBEDDING_MODEL)
    except FileNotFoundError:
        pass
    yield


app = FastAPI(
    title="IR Retrieval Service",
    description="Library-based retrieval: sklearn TF-IDF, rank_bm25, offline FAISS, SQLite documents.",
    version="3.0.0",
    lifespan=lifespan,
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    from shared.config import SERVICE_PORTS

    uvicorn.run(
        "retrieval_service.main:app",
        host="127.0.0.1",
        port=SERVICE_PORTS["retrieval"],
        reload=False,
    )
