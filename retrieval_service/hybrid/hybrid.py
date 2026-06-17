from retrieval_service.hybrid.serial import SerialHybrid
from retrieval_service.hybrid.parallel import ParallelHybrid


class HybridStrategy:

    def __init__(self, mode="serial"):

        self.serial = SerialHybrid()
        self.parallel = ParallelHybrid()
        self.mode = mode

    def fit(self, documents):
        self.serial.fit(documents)
        self.parallel.fit(documents)

    def search(self, query):

        if self.mode == "serial":
            return self.serial.search(query)

        return self.parallel.search(query)