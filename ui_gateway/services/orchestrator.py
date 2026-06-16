"""
This file orchestrates the UI request.
It gathers tokens from the preprocessing service, retrieves results from the retrieval service,
and packages them safely for the Jinja2 template view.
"""
from shared.schemas import SearchRequest, RetrievalModel
from ui_gateway.infrastructure.backend_client import BackendAPIClient

class UIGatewayOrchestrator:
    def __init__(self) -> None:
        self.api_client = BackendAPIClient()

    async def handle_search(self, query: str, model_str: str, top_k: int) -> dict:
        context = {
            "query": query,
            "model": model_str,
            "k": top_k,
            "error": None,
            "query_tokens": [],
            "results": []
        }

        try:
            # Map HTML form string to Enum
            model_enum = self._map_model_string(model_str)
            
            # 1. Fetch Tokens for UI display
            context["query_tokens"] = await self.api_client.get_processed_tokens(query)

            # 2. Execute Search
            search_req = SearchRequest(query=query, model=model_enum, top_k=top_k)
            search_response = await self.api_client.execute_search(search_req)
            
            # Bind results back to view context
            context["results"] = search_response.results
            context["elapsed_ms"] = search_response.elapsed_ms

        except Exception as e:
            context["error"] = f"Failed to fetch results: {str(e)}"

        return context

    @staticmethod
    def _map_model_string(model_str: str) -> RetrievalModel:
        mapping = {
            "tfidf": RetrievalModel.TF_IDF,
            "bm25": RetrievalModel.BM25,
            "hybrid": RetrievalModel.HYBRID_PARALLEL
        }
        return mapping.get(model_str, RetrievalModel.TF_IDF)