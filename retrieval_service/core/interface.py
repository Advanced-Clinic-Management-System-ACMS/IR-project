from abc import ABC, abstractmethod

class RetrievalStrategy(ABC):

    @abstractmethod
    def fit(self, documents):
        pass

    @abstractmethod
    def search(self, query):
        pass