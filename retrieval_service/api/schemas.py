from pydantic import BaseModel
from typing import List


class SearchRequest(BaseModel):
    model_type: str
    query: str
    documents: List[str]
    mode: str = "serial"


class SearchResponse(BaseModel):
    results: list