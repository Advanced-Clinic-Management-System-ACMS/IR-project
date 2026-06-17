from retrieval_service.core.interface import RetrievalStrategy
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np

class TFIDFStrategy(RetrievalStrategy):

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.matrix = None

    def fit(self, documents):
        self.docs = documents
        self.matrix = self.vectorizer.fit_transform(documents)

    def search(self, query):
        q_vec = self.vectorizer.transform([query])
        scores = (self.matrix @ q_vec.T).toarray().ravel()

        return list(enumerate(scores))