from fastapi import FastAPI, HTTPException

from retrieval_service.retriever import RetrievalEngine
from shared.database import DocumentStore
from shared.schemas import HealthResponse, SearchRequest, SearchResponse, SearchResultItem

app = FastAPI(
    title="Retrieval Service",
    description="Search using TF-IDF, BM25, Embeddings, and Hybrid fusion.",
    version="1.0.0",
)

engine = RetrievalEngine()
document_store = DocumentStore()


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    mongo_ok = False
    try:
        mongo_ok = document_store.ping()
    except Exception:
        mongo_ok = False

    return HealthResponse(
        service="retrieval_service",
        status="ok",
        details={"mongodb_connected": mongo_ok},
    )


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    try:
        ranked, elapsed_ms = engine.search(
            query=request.query,
            model=request.model.value,
            top_k=request.top_k,
            bm25_k1=request.bm25_k1,
            bm25_b=request.bm25_b,
            hybrid_weights=request.hybrid_weights,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    doc_ids = [doc_id for doc_id, _ in ranked]
    raw_docs = document_store.get_documents_by_ids(doc_ids)
    raw_map = {doc["doc_id"]: doc for doc in raw_docs}

    results: list[SearchResultItem] = []
    for rank, (doc_id, score) in enumerate(ranked, start=1):
        raw_doc = raw_map.get(doc_id)
        snippet = None
        if raw_doc:
            text = raw_doc.get("text", "")
            snippet = text[:240] + ("..." if len(text) > 240 else "")

        results.append(
            SearchResultItem(
                doc_id=doc_id,
                score=round(score, 6),
                rank=rank,
                snippet=snippet,
            )
        )

    return SearchResponse(
        query=request.query,
        model=request.model,
        results=results,
        elapsed_ms=round(elapsed_ms, 3),
    )


if __name__ == "__main__":
    import uvicorn

    from shared.config import SERVICE_PORTS

    uvicorn.run(
        "retrieval_service.main:app",
        host="127.0.0.1",
        port=SERVICE_PORTS["retrieval"],
        reload=True,
    )
