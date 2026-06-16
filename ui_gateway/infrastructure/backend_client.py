"""
This file acts as the HTTP client for the UI Gateway to communicate with backend microservices.
It strictly enforces types and handles connection timeouts.
"""
import httpx
from shared.config import SERVICE_URLS
from shared.schemas import SearchRequest, SearchResponse

class BackendAPIClient:
    def __init__(self) -> None:
        self.retrieval_url = SERVICE_URLS["retrieval"]
        self.preprocessing_url = SERVICE_URLS["preprocessing"]

    async def get_processed_tokens(self, query: str) -> list[str]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{self.preprocessing_url}/process-query", 
                params={"query_text": query}
            )
            response.raise_for_status()
            return response.json().get("tokens", [])

    async def execute_search(self, request: SearchRequest) -> SearchResponse:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.retrieval_url}/search", 
                content=request.model_dump_json()
            )
            response.raise_for_status()
            return SearchResponse(**response.json())