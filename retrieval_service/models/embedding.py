from retrieval_service.core.interface import RetrievalStrategy
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingStrategy(RetrievalStrategy):

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def fit(self, documents):
        self.docs = documents
        self.doc_embeddings = self.model.encode(documents)

    def search(self, query):
        q = self.model.encode([query])[0]

        scores = np.dot(self.doc_embeddings, q)

        return list(enumerate(scores))