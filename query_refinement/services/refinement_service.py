"""
This file orchestrates the query refinement use case.
It coordinates the input schemas and the core NLP logic to generate refined queries.
"""
from query_refinement.schemas.payloads import RefineRequest, RefineResponse
from query_refinement.core.refiner import QueryRefinerCore


class RefinementOrchestrator:
    def __init__(self) -> None:
        self.core = QueryRefinerCore()

    def execute_refinement(self, request: RefineRequest) -> RefineResponse:
        original = self.core.normalize_query(request.query)
        refined_query, refined_tokens = self.core.refine(original)
        original_tokens = self.core.tokenize_query(original)

        suggestions = []
        if request.history:
            suggestions.append(f"Related to your last search: {request.history[-1]}")

        if refined_tokens != original_tokens:
            suggestions.append("Did you mean: " + " ".join(refined_tokens[: len(original_tokens) + 2]))

        if len(refined_tokens) > len(original_tokens):
            suggestions.append("Expanded with related terms for broader recall.")

        return RefineResponse(
            original_query=original,
            refined_query=refined_query,
            suggestions=suggestions,
        )
