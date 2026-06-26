"""
HTTP client for the query refinement microservice.
"""
from __future__ import annotations

import requests

from shared.config import SERVICE_URLS


class RefinementClient:
    def __init__(self, base_url: str | None = None, timeout: float = 15.0) -> None:
        self.base_url = base_url or SERVICE_URLS["query_refinement"]
        self.timeout = timeout

    def refine_query(
        self,
        query_text: str,
        history: list[str] | None = None,
        use_refinement: bool = True,
        use_personalization: bool = False,
    ) -> tuple[str, list[str]]:
        response = requests.post(
            f"{self.base_url}/refine",
            json={
                "query": query_text,
                "history": history or [],
                "use_refinement": use_refinement,
                "use_personalization": use_personalization,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return (
            payload.get("refined_query", query_text),
            payload.get("personalization_applied", []),
        )
