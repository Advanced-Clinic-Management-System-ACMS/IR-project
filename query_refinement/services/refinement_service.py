"""
Orchestrates query refinement and personalization (Requirement 5 + 16).
Each feature can be toggled independently for demo and evaluation.
"""
from query_refinement.core.personalization import PersonalizationEngine
from query_refinement.core.refiner import QueryRefinerCore
from query_refinement.schemas.payloads import RefineRequest, RefineResponse


class RefinementOrchestrator:
    def __init__(self) -> None:
        self.core = QueryRefinerCore()
        self.personalization = PersonalizationEngine()

    def execute_refinement(self, request: RefineRequest) -> RefineResponse:
        original = self.core.normalize_query(request.query)
        original_tokens = self.core.tokenize_query(original)

        if request.use_refinement:
            refined_query, refined_tokens = self.core.refine(original)
        else:
            refined_query = original
            refined_tokens = list(original_tokens)

        suggestions: list[str] = []
        personalization_applied: list[str] = []

        if request.use_personalization and request.history:
            personalized_tokens, personalization_applied = self.personalization.apply_history(
                refined_tokens,
                request.history,
            )
            refined_tokens = self.core.deduplicate_tokens(personalized_tokens)
            refined_query = " ".join(refined_tokens)
            suggestions.append(
                "Personalization applied: " + ", ".join(personalization_applied)
            )

        if request.use_refinement and refined_tokens != original_tokens:
            suggestions.append("Did you mean: " + " ".join(refined_tokens[: len(original_tokens) + 2]))

        if request.use_refinement and len(refined_tokens) > len(original_tokens):
            suggestions.append("Query expanded with related terms for broader recall.")

        return RefineResponse(
            original_query=original,
            refined_query=refined_query,
            suggestions=suggestions,
            personalization_applied=personalization_applied,
        )
