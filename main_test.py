"""Quick retrieval engine smoke test against built indexes."""
from shared.schemas import RetrievalModel, SearchRequest
from retrieval_service.services.engine import RetrievalEngine

engine = RetrievalEngine()
response = engine.search(
    SearchRequest(
        query="information retrieval system",
        model=RetrievalModel.TF_IDF,
        top_k=3,
    )
)
print(f"Query: {response.query}")
print(f"Elapsed: {response.elapsed_ms} ms")
for item in response.results:
    print(f"  [{item.rank}] {item.doc_id} score={item.score}")
    print(f"      {((item.text or '')[:120])}...")
