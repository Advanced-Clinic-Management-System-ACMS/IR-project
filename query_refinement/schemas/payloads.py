"""
This file defines the strict data contracts for the Query Refinement service.
"""
from pydantic import BaseModel, Field


class RefineRequest(BaseModel):
    query: str = Field(..., description="The raw search query from the user")
    history: list[str] = Field(default_factory=list, description="Previous queries in the current session")
    use_refinement: bool = Field(default=True, description="Apply spelling correction and synonym expansion")
    use_personalization: bool = Field(default=False, description="Apply history-based personalization (#16)")


class RefineResponse(BaseModel):
    original_query: str
    refined_query: str
    suggestions: list[str] = Field(default_factory=list)
    personalization_applied: list[str] = Field(default_factory=list)
