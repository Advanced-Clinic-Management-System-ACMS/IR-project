from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RetrievalModel(str, Enum):
    TF_IDF = "tf_idf"
    BM25 = "bm25"
    EMBEDDING = "embedding"
    HYBRID_SERIAL = "hybrid_serial"
    HYBRID_PARALLEL = "hybrid_parallel"
    HYBRID_BRANCHING = "hybrid_branching"


class DocumentInput(BaseModel):
    doc_id: str
    text: str
    title: str | None = None


class ProcessedDocument(BaseModel):
    doc_id: str
    tokens: list[str]
    original_text: str


class PreprocessRequest(BaseModel):
    documents: list[DocumentInput]


class PreprocessResponse(BaseModel):
    processed: list[ProcessedDocument]
    count: int


class BuildIndexRequest(BaseModel):
    processed_documents: list[ProcessedDocument]
    dataset_name: str = "default"
    save_embeddings: bool = False


class BuildIndexResponse(BaseModel):
    dataset_name: str
    document_count: int
    vocabulary_size: int
    index_path: str
    message: str


class SearchRequest(BaseModel):
    query: str
    model: RetrievalModel = RetrievalModel.TF_IDF
    top_k: int = Field(default=10, ge=1, le=100)
    dataset_name: str = "lotte_lifestyle_dev_forum"
    bm25_k1: float | None = None
    bm25_b: float | None = None
    hybrid_weights: dict[str, float] | None = None
    fusion_mode: str = "rrf"
    use_refinement: bool = False
    use_personalization: bool = False
    user_history: list[str] = Field(default_factory=list)


class SearchResultItem(BaseModel):
    doc_id: str
    score: float
    rank: int
    snippet: str | None = None
    text: str | None = None
    title: str | None = None


class SearchResponse(BaseModel):
    query: str
    original_query: str | None = None
    model: RetrievalModel
    results: list[SearchResultItem]
    elapsed_ms: float
    dataset_name: str | None = None
    query_tokens: list[str] = Field(default_factory=list)
    use_refinement: bool = False
    use_personalization: bool = False
    personalization_applied: list[str] = Field(default_factory=list)
    fusion_mode: str | None = None


class HealthResponse(BaseModel):
    service: str
    status: str
    details: dict[str, Any] | None = None
