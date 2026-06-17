from fastapi import APIRouter

from retrieval_service.api.schemas import SearchRequest, SearchResponse
from retrieval_service.engine.search_engine import SearchEngine


router = APIRouter()
engine = SearchEngine()


@router.get("/")
def home():
    return {"message": "IR System Running"}


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):

    results = engine.search(
        request.model_type,
        request.query,
        request.documents,
        request.mode
    )

    return SearchResponse(results=results)