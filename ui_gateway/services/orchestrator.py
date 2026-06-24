"""
UI request orchestrator.
"""
from __future__ import annotations

from shared.config import DEFAULT_BM25_B, DEFAULT_BM25_K1, DEFAULT_DATASET_NAME
from shared.schemas import RetrievalModel, SearchRequest
from ui_gateway.infrastructure.backend_client import BackendAPIClient

import httpx


class UIGatewayOrchestrator:
    def __init__(self) -> None:
        self.api_client = BackendAPIClient()

    async def handle_search(
        self,
        query: str,
        model_str: str,
        top_k: int,
        bm25_k1: float,
        bm25_b: float,
        use_refinement: bool,
        dataset_name: str = DEFAULT_DATASET_NAME,
        user_history: str = "",
        fusion_mode: str = "rrf",
        weight_tfidf: float = 0.34,
        weight_bm25: float = 0.33,
        weight_embedding: float = 0.33,
    ) -> dict:
        context = {
            "query": query,
            "model": model_str,
            "k": top_k,
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "use_refinement": use_refinement,
            "dataset_name": dataset_name,
            "user_history": user_history,
            "fusion_mode": fusion_mode,
            "weight_tfidf": weight_tfidf,
            "weight_bm25": weight_bm25,
            "weight_embedding": weight_embedding,
            "error": None,
            "query_tokens": [],
            "refined_query": None,
            "personalization_applied": [],
            "suggestions": [],
            "results": [],
            "elapsed_ms": None,
        }

        try:
            model_enum = self._map_model_string(model_str)
            history_list = [h.strip() for h in user_history.split(",") if h.strip()]

            search_req = SearchRequest(
                query=query,
                model=model_enum,
                top_k=top_k,
                dataset_name=dataset_name,
                bm25_k1=bm25_k1,
                bm25_b=bm25_b,
                use_refinement=use_refinement,
                user_history=history_list,
                fusion_mode=fusion_mode,
                hybrid_weights={
                    "tfidf": weight_tfidf,
                    "bm25": weight_bm25,
                    "embedding": weight_embedding,
                },
            )
            search_response = await self.api_client.execute_search(search_req)
            context["results"] = search_response.results
            context["elapsed_ms"] = search_response.elapsed_ms
            context["query_tokens"] = search_response.query_tokens
            context["personalization_applied"] = search_response.personalization_applied
            if search_response.query != query:
                context["refined_query"] = search_response.query

        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            try:
                detail = exc.response.json().get("detail", detail)
            except Exception:
                pass
            context["error"] = f"Backend error ({exc.response.status_code}): {detail}"
        except Exception as exc:
            context["error"] = f"Failed to fetch results: {exc}"

        return context

    @staticmethod
    def _map_model_string(model_str: str) -> RetrievalModel:
        mapping = {
            "tfidf": RetrievalModel.TF_IDF,
            "bm25": RetrievalModel.BM25,
            "embedding": RetrievalModel.EMBEDDING,
            "hybrid_serial": RetrievalModel.HYBRID_SERIAL,
            "hybrid_parallel": RetrievalModel.HYBRID_PARALLEL,
            "hybrid_branching": RetrievalModel.HYBRID_BRANCHING,
            "hybrid": RetrievalModel.HYBRID_PARALLEL,
        }
        return mapping.get(model_str, RetrievalModel.TF_IDF)
