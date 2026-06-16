"""
This file acts as an HTTP client to communicate with the Preprocessing Service.
It enforces strict SOA boundaries by ensuring inter-service communication over the network.
"""
import requests
from shared.config import SERVICE_URLS

class PreprocessingClient:
    def __init__(self) -> None:
        self.base_url = SERVICE_URLS["preprocessing"]

    def get_query_tokens(self, query_text: str) -> list[str]:
        try:
            response = requests.post(f"{self.base_url}/process-query", params={"query_text": query_text})
            response.raise_for_status()
            data = response.json()
            return data.get("tokens", [])
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to communicate with Preprocessing Service: {str(e)}")