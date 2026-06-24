"""
Application orchestrator for retrieval.
"""
from __future__ import annotations

import time

from shared.config import DEFAULT_BM25_B, DEFAULT_BM25_K1, DEFAULT_DATASET_NAME
from shared.schemas import SearchRequest, SearchResponse, SearchResultItem
from retrieval_service.core.factory import RetrievalFactory
from retrieval_service.core.scoring import rank_documents
from retrieval_service.infrastructure.document_repo import DocumentRepository
from retrieval_service.infrastructure.index_repo import IndexRepository
from retrieval_service.infrastructure.preprocess_client import PreprocessClient
from retrieval_service.infrastructure.refinement_client import RefinementClient


class RetrievalEngine:
    def __init__(self) -> None:
        self.preprocess_client = PreprocessClient()
        self.refinement_client = RefinementClient()

    def search(self, request: SearchRequest) -> SearchResponse:
        started = time.perf_counter()
        dataset_name = request.dataset_name or DEFAULT_DATASET_NAME
        index_repo = IndexRepository(dataset_name)
        loaded_index = index_repo.load()
        fusion_mode = request.fusion_mode or "rrf"
        index_data = index_repo.as_scoring_payload(loaded_index, fusion_mode=fusion_mode)
        document_repo = DocumentRepository(dataset_name)

        query_text = request.query
        personalization_applied: list[str] = []
        history = request.user_history or []

        if request.use_refinement or history:
            query_text, personalization_applied = self.refinement_client.refine_query(
                request.query,
                history=history,
            )

        query_tokens = self.preprocess_client.process_query(query_text)
        index_data["query_text"] = query_text
        k1 = request.bm25_k1 if request.bm25_k1 is not None else DEFAULT_BM25_K1
        b = request.bm25_b if request.bm25_b is not None else DEFAULT_BM25_B

        strategy = RetrievalFactory.create(request.model)
        scores = strategy.score(
            query_tokens,
            index_data,
            k1=k1,
            b=b,
            weights=request.hybrid_weights,
        )
        ranked = rank_documents(scores, request.top_k)
        doc_ids = [doc_id for doc_id, _ in ranked]
        raw_docs = document_repo.get_documents_by_ids(doc_ids)

        results: list[SearchResultItem] = []
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            raw = raw_docs.get(doc_id, {})
            text = raw.get("text") or ""
            results.append(
                SearchResultItem(
                    doc_id=doc_id,
                    score=round(score, 6),
                    rank=rank,
                    text=text or None,
                    title=raw.get("title"),
                    snippet=text[:300] or None,
                )
            )

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return SearchResponse(
            query=query_text,
            original_query=request.query,
            model=request.model,
            results=results,
            elapsed_ms=elapsed_ms,
            dataset_name=dataset_name,
            query_tokens=query_tokens,
            use_refinement=request.use_refinement,
            personalization_applied=personalization_applied,
            fusion_mode=fusion_mode,
        )
