from retrieval_service.core.factory import StrategyFactory


class SearchEngine:

    def search(self, model_type, query, documents, mode="serial"):

        strategy = StrategyFactory.create(model_type, mode)

        strategy.fit(documents)

        return strategy.search(query)