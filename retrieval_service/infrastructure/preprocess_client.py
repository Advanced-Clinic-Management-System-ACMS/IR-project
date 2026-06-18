"""
HTTP client for the preprocessing microservice.
Enforces SOA boundary — retrieval never imports preprocessing code directly.
"""
from __future__ import annotations

import requests

from shared.config import SERVICE_URLS


class PreprocessClient:
    def __init__(self, base_url: str | None = None, timeout: float = 30.0) -> None:
        self.base_url = base_url or SERVICE_URLS["preprocessing"]
        self.timeout = timeout

    def process_query(self, query_text: str) -> list[str]:
        response = requests.post(
            f"{self.base_url}/process-query",
            params={"query_text": query_text},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("tokens", [])
