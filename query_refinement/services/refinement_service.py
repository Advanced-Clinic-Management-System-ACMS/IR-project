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
        # 1. Basic tokenization for refinement
        tokens = request.query.split()

        # 2. Apply core logic
        corrected_tokens = self.core.correct_spelling(tokens)
        expanded_tokens = self.core.expand_synonyms(corrected_tokens)

        # 3. Construct refined query
        refined_query = " ".join(expanded_tokens)

        # 4. Generate intelligent suggestions
        suggestions = []
        if request.history:
            suggestions.append(f"Related to your last search: {request.history[-1]}")
        
        if corrected_tokens != tokens:
            suggestions.append("Did you mean: " + " ".join(corrected_tokens))

        return RefineResponse(
            original_query=request.query,
            refined_query=refined_query,
            suggestions=suggestions
        )