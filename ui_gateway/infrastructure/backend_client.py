"""
HTTP client for the UI Gateway to communicate with backend microservices.
"""
from __future__ import annotations

import httpx

from shared.config import DEFAULT_DATASET_NAME, SERVICE_URLS
from shared.schemas import SearchRequest, SearchResponse


class BackendAPIClient:
    def __init__(self) -> None:
        self.retrieval_url = SERVICE_URLS["retrieval"]
        self.preprocessing_url = SERVICE_URLS["preprocessing"]
        self.refinement_url = SERVICE_URLS["query_refinement"]

    async def get_processed_tokens(self, query: str) -> list[str]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{self.preprocessing_url}/process-query",
                params={"query_text": query},
            )
            response.raise_for_status()
            return response.json().get("tokens", [])

    # Add 'history' to the arguments
    async def refine_query(self, query: str, history: list[str] = None) -> str:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{self.refinement_url}/refine",
                # Pass the history variable instead of []
                json={"query": query, "history": history or []},
            )
            response.raise_for_status()
            return response.json().get("refined_query", query)

    async def execute_search(self, request: SearchRequest) -> SearchResponse:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.retrieval_url}/search",
                json=request.model_dump(mode="json"),
            )
            response.raise_for_status()
            return SearchResponse(**response.json())
