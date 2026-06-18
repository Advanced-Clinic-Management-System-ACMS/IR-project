from retrieval_service.core.scoring import score_bm25
from retrieval_service.core.strategies.base import RetrievalStrategy


class BM25Strategy(RetrievalStrategy):
    def score(
        self,
        query_tokens: list[str],
        index_data: dict,
        k1: float,
        b: float,
        weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        return score_bm25(
            query_tokens,
            index_data["inverted_index"],
            index_data["doc_lengths"],
            index_data["avg_doc_length"],
            index_data["idf"],
            k1=k1,
            b=b,
        )
