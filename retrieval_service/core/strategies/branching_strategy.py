from retrieval_service.core.scoring import compute_all_lexical_scores, fuse_branching
from retrieval_service.core.strategies.base import RetrievalStrategy


class BranchingHybridStrategy(RetrievalStrategy):
    def score(
        self,
        query_tokens: list[str],
        index_data: dict,
        k1: float,
        b: float,
        weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        tfidf_scores, bm25_scores, embedding_scores = compute_all_lexical_scores(
            query_tokens, index_data, k1, b
        )
        return fuse_branching(query_tokens, tfidf_scores, bm25_scores, embedding_scores)
