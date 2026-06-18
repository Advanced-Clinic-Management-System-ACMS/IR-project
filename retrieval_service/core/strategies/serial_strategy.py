from retrieval_service.core.scoring import fuse_serial_rerank
from retrieval_service.core.strategies.base import RetrievalStrategy


class SerialHybridStrategy(RetrievalStrategy):
    def score(
        self,
        query_tokens: list[str],
        index_data: dict,
        k1: float,
        b: float,
        weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        return fuse_serial_rerank(query_tokens, index_data, k1, b)
