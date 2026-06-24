from retrieval_service.core.scoring import fuse_rrf, fuse_weighted, run_parallel_scorers
from retrieval_service.core.strategies.base import RetrievalStrategy


class ParallelHybridStrategy(RetrievalStrategy):
    def score(
        self,
        query_tokens: list[str],
        index_data: dict,
        k1: float,
        b: float,
        weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        tfidf_scores, bm25_scores, embedding_scores = run_parallel_scorers(
            query_tokens, index_data, k1, b
        )
        if index_data.get("fusion_mode") == "weighted":
            return fuse_weighted(tfidf_scores, bm25_scores, embedding_scores, weights)
        return fuse_rrf([tfidf_scores, bm25_scores, embedding_scores])
