from abc import ABC, abstractmethod


class RetrievalStrategy(ABC):
    @abstractmethod
    def score(
        self,
        query_tokens: list[str],
        index_data: dict,
        k1: float,
        b: float,
        weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Return document_id -> relevance score."""
