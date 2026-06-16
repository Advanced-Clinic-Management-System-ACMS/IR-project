"""
This file orchestrates the retrieval use case.
It coordinates the preprocessing client, the index repository, the scoring math, 
and the MongoDB repository to generate the final search results.
"""
import time
from shared.config import DEFAULT_BM25_B, DEFAULT_BM25_K1
from shared.schemas import SearchRequest, SearchResultItem, RetrievalModel
from retrieval_service.core.scoring import ScoringCore
from retrieval_service.infrastructure.index_repo import IndexRepository
from retrieval_service.infrastructure.mongo_repo import MongoDocumentRepository
from retrieval_service.infrastructure.preprocess_client import PreprocessingClient

class RetrievalEngineService:
    def __init__(self, dataset_name: str = "default") -> None:
        self.index_repo = IndexRepository(dataset_name)
        self.mongo_repo = MongoDocumentRepository()
        self.preprocess_client = PreprocessingClient()
        self.index_data = self.index_repo.load_index_data()

    def execute_search(self, request: SearchRequest) -> tuple[list[SearchResultItem], float]:
        start = time.perf_counter()
        
        # 1. Get Tokens via HTTP Client
        query_tokens = self.preprocess_client.get_query_tokens(request.query)

        # 2. Execute Math/Scoring
        scores = {}
        if request.model == RetrievalModel.TF_IDF:
            scores = ScoringCore.compute_tf_idf(query_tokens, self.index_data["inverted_index"], self.index_data["tf_idf_vectors"], self.index_data["bm25_stats"]["idf"])
        elif request.model == RetrievalModel.BM25:
            scores = ScoringCore.compute_bm25(query_tokens, self.index_data["inverted_index"], self.index_data["bm25_stats"], self.index_data["doc_lengths"], request.bm25_k1 or DEFAULT_BM25_K1, request.bm25_b or DEFAULT_BM25_B)
        elif request.model == RetrievalModel.EMBEDDING:
            if self.index_data["embeddings"] is None:
                raise ValueError("Embeddings were not built for this dataset.")
            vocabulary = list(self.index_data["bm25_stats"]["doc_freq"].keys())
            scores = ScoringCore.compute_embedding(query_tokens, self.index_data["embeddings"], self.index_data["embedding_doc_ids"], vocabulary)
        
        # Sort and get Top K
        ranked_doc_ids = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:request.top_k]
        
        # 3. Fetch Snippets from MongoDB
        doc_ids = [doc_id for doc_id, _ in ranked_doc_ids]
        raw_docs = self.mongo_repo.get_documents_by_ids(doc_ids)

        # 4. Construct strictly typed output
        results = []
        for rank, (doc_id, score) in enumerate(ranked_doc_ids, start=1):
            snippet = raw_docs.get(doc_id, {}).get("text", "")[:200] + "..."  # Short snippet
            results.append(
                SearchResultItem(doc_id=doc_id, score=score, rank=rank, snippet=snippet)
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        return results, elapsed_ms