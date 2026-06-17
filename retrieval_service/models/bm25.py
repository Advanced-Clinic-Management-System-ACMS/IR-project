from retrieval_service.core.interface import RetrievalStrategy
from rank_bm25 import BM25Okapi

class BM25Strategy(RetrievalStrategy):

    def __init__(self):
        self.bm25 = None
        self.docs = None

    def fit(self, documents):
        self.docs = documents
        tokenized = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        return list(enumerate(scores))