"""
This file defines strict Pydantic schemas specific to the preprocessing service.
It ensures that all API responses, even for simple queries, are strongly typed 
and validated before leaving the service.
"""
from pydantic import BaseModel

class QueryProcessResponse(BaseModel):
    query_id: str
    original_text: str
    tokens: list[str]
    token_count: int