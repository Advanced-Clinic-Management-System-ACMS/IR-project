import numpy as np
from concurrent.futures import ThreadPoolExecutor

from retrieval_service.models.tfidf import TFIDFStrategy
from retrieval_service.models.bm25 import BM25Strategy
from retrieval_service.models.embedding import EmbeddingStrategy


class ParallelHybrid:

    def __init__(self):
        self.tfidf = TFIDFStrategy()
        self.bm25 = BM25Strategy()
        self.embed = EmbeddingStrategy()

    def fit(self, documents):
        self.tfidf.fit(documents)
        self.bm25.fit(documents)
        self.embed.fit(documents)

    def _run(self, model, query):
        return model.search(query)

    def search(self, query):

        with ThreadPoolExecutor() as ex:

            r1 = ex.submit(self._run, self.tfidf, query)
            r2 = ex.submit(self._run, self.bm25, query)
            r3 = ex.submit(self._run, self.embed, query)

            results = r1.result() + r2.result() + r3.result()

        final = {}

        for i, s in results:
            final[i] = final.get(i, 0) + float(s)

        return sorted(final.items(), key=lambda x: x[1], reverse=True)