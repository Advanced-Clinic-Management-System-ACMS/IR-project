from shared.schemas import RetrievalModel
from retrieval_service.core.strategies import (
    BM25Strategy,
    BranchingHybridStrategy,
    EmbeddingStrategy,
    ParallelHybridStrategy,
    RetrievalStrategy,
    SerialHybridStrategy,
    TFIDFStrategy,
)


class RetrievalFactory:
    @staticmethod
    def create(model: RetrievalModel) -> RetrievalStrategy:
        mapping: dict[RetrievalModel, RetrievalStrategy] = {
            RetrievalModel.TF_IDF: TFIDFStrategy(),
            RetrievalModel.BM25: BM25Strategy(),
            RetrievalModel.EMBEDDING: EmbeddingStrategy(),
            RetrievalModel.HYBRID_SERIAL: SerialHybridStrategy(),
            RetrievalModel.HYBRID_PARALLEL: ParallelHybridStrategy(),
            RetrievalModel.HYBRID_BRANCHING: BranchingHybridStrategy(),
        }
        strategy = mapping.get(model)
        if strategy is None:
            raise ValueError(f"Unsupported retrieval model: {model}")
        return strategy
