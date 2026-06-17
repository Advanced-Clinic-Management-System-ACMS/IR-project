from retrieval_service.engine.search_engine import SearchEngine
engine = SearchEngine()

docs = [
    "machine learning is amazing",
    "deep learning improves search",
    "information retrieval uses bm25",
    "bert embeddings are powerful",
    "nlp and search systems"
]

print(engine.search("tfidf", "machine learning", docs))
print(engine.search("bm25", "information retrieval", docs))
print(engine.search("embedding", "deep learning", docs))
print(engine.search("hybrid_serial", "search systems", docs))
print(engine.search("hybrid_parallel", "search systems", docs))