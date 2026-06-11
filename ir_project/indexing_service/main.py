from fastapi import FastAPI, HTTPException

from indexing_service.indexer import InvertedIndexBuilder
from shared.schemas import BuildIndexRequest, BuildIndexResponse, HealthResponse

app = FastAPI(
    title="Indexing Service",
    description="Build inverted index, TF-IDF weights, BM25 stats, and optional embeddings.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(service="indexing_service", status="ok")


@app.post("/build-index", response_model=BuildIndexResponse)
def build_index(request: BuildIndexRequest) -> BuildIndexResponse:
    if not request.processed_documents:
        raise HTTPException(status_code=400, detail="processed_documents cannot be empty.")

    builder = InvertedIndexBuilder(request.dataset_name)
    metadata = builder.build(
        request.processed_documents,
        save_embeddings=request.save_embeddings,
    )

    return BuildIndexResponse(
        dataset_name=metadata["dataset_name"],
        document_count=metadata["document_count"],
        vocabulary_size=metadata["vocabulary_size"],
        index_path=metadata["index_path"],
        message="Index built successfully.",
    )


if __name__ == "__main__":
    import uvicorn

    from shared.config import SERVICE_PORTS

    uvicorn.run(
        "indexing_service.main:app",
        host="127.0.0.1",
        port=SERVICE_PORTS["indexing"],
        reload=True,
    )
