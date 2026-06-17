from retrieval_service.models.tfidf import TFIDFStrategy
from retrieval_service.models.bm25 import BM25Strategy
from retrieval_service.models.embedding import EmbeddingStrategy
from retrieval_service.hybrid.hybrid import HybridStrategy


class StrategyFactory:

    @staticmethod
    def create(model_type, mode="serial"):

        if model_type == "tfidf":
            return TFIDFStrategy()

        if model_type == "bm25":
            return BM25Strategy()

        if model_type == "embedding":
            return EmbeddingStrategy()

        if model_type == "hybrid":
            return HybridStrategy(mode)

        raise Exception("Unknown model type")