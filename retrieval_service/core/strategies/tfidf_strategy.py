from retrieval_service.core.scoring import score_tfidf
from retrieval_service.core.strategies.base import RetrievalStrategy


class TFIDFStrategy(RetrievalStrategy):
    def score(
        self,
        query_tokens: list[str],
        index_data: dict,
        k1: float,
        b: float,
        weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        return score_tfidf(query_tokens, index_data["tf_idf_vectors"], index_data["idf"])
