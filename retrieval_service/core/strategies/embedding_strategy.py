from retrieval_service.core.scoring import score_embedding_semantic
from retrieval_service.core.strategies.base import RetrievalStrategy


class EmbeddingStrategy(RetrievalStrategy):
    def score(
        self,
        query_tokens: list[str],
        index_data: dict,
        k1: float,
        b: float,
        weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        return score_embedding_semantic(
            query_tokens,
            index_data.get("query_text", ""),
            index_data,
        )
