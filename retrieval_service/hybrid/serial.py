import numpy as np

from retrieval_service.models.tfidf import TFIDFStrategy
from retrieval_service.models.bm25 import BM25Strategy
from retrieval_service.models.embedding import EmbeddingStrategy


class SerialHybrid:

    def __init__(self):
        self.tfidf = TFIDFStrategy()
        self.bm25 = BM25Strategy()
        self.embed = EmbeddingStrategy()

    def fit(self, documents):
        self.tfidf.fit(documents)
        self.bm25.fit(documents)
        self.embed.fit(documents)

    def search(self, query):

        s1 = self.tfidf.search(query)
        s2 = self.bm25.search(query)
        s3 = self.embed.search(query)

        final = {}

        for i, s in s1 + s2 + s3:
            final[i] = final.get(i, 0) + float(s)

        return sorted(final.items(), key=lambda x: x[1], reverse=True)