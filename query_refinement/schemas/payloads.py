"""
This file defines the strict data contracts for the Query Refinement service.
It replaces generic dictionaries with statically analyzed Pydantic models.
"""
from pydantic import BaseModel, Field

class RefineRequest(BaseModel):
    query: str = Field(..., description="The raw search query from the user")
    history: list[str] = Field(default_factory=list, description="Previous queries in the current session")

class RefineResponse(BaseModel):
    original_query: str
    refined_query: str
    suggestions: list[str]