# retrieval_service/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import json
import math
from pathlib import Path
from collections import Counter

from shared.config import INDEX_DIR
from shared.schemas import HealthResponse

app = FastAPI(title="Retrieval Service", version="1.0.0")

class SearchRequest(BaseModel):
    dataset_name: str
    query_tokens: List[str]
    model_type: str  # "tfidf", "bm25", "hybrid"
    k: int = Field(default=10, ge=1, le=100)

class SearchResponse(BaseModel):
    query_id: str
    results: List[Dict[str, Any]]
    model_type: str

class RetrievalService:
    def __init__(self):
        self.indexes = {}
        
    def load_index(self, dataset_name: str):
        if dataset_name in self.indexes:
            return self.indexes[dataset_name]
        
        dataset_dir = INDEX_DIR / dataset_name
        
        try:
            with open(dataset_dir / "inverted_index.json", encoding='utf-8') as f:
                inverted_index = json.load(f)
            
            with open(dataset_dir / "idf.json", encoding='utf-8') as f:
                idf = json.load(f)
                
            with open(dataset_dir / "bm25_stats.json", encoding='utf-8') as f:
                bm25_stats = json.load(f)
                
            with open(dataset_dir / "tf_idf_vectors.json", encoding='utf-8') as f:
                tf_idf_vectors = json.load(f)
            
            # تحويل doc_lengths من string keys إلى int keys إذا لزم
            if "doc_lengths" in bm25_stats:
                doc_lengths = {}
                for k, v in bm25_stats["doc_lengths"].items():
                    doc_lengths[str(k)] = v
                bm25_stats["doc_lengths"] = doc_lengths
            
            self.indexes[dataset_name] = {
                "inverted_index": inverted_index,
                "idf": idf,
                "bm25_stats": bm25_stats,
                "tf_idf_vectors": tf_idf_vectors,
                "doc_ids": list(tf_idf_vectors.keys())
            }
        except FileNotFoundError as e:
            print(f"⚠️ لم يتم العثور على الفهرس للمجموعة {dataset_name}")
            print(f"يرجى تشغيل indexing_service أولاً لبناء الفهرس")
            raise
        
        return self.indexes[dataset_name]
    
    def compute_tfidf_score(self, query_tokens: List[str], doc_id: str, index_data: dict) -> float:
        """حساب تشابه TF-IDF"""
        query_tf = Counter(query_tokens)
        query_vector = {}
        
        # بناء متجه الاستعلام
        for term, tf in query_tf.items():
            if term in index_data["idf"]:
                query_vector[term] = (tf / len(query_tokens)) * index_data["idf"][term]
        
        # جلب متجه الوثيقة
        doc_vector = index_data["tf_idf_vectors"].get(doc_id, {})
        
        if not query_vector or not doc_vector:
            return 0.0
        
        # حساب جيب التمام
        dot_product = 0.0
        query_norm = 0.0
        doc_norm = 0.0
        
        for term, q_weight in query_vector.items():
            query_norm += q_weight ** 2
            if term in doc_vector:
                dot_product += q_weight * doc_vector[term]
        
        for d_weight in doc_vector.values():
            doc_norm += d_weight ** 2
        
        if query_norm == 0 or doc_norm == 0:
            return 0.0
        
        return dot_product / (math.sqrt(query_norm) * math.sqrt(doc_norm))
    
    def compute_bm25_score(self, query_tokens: List[str], doc_id: str, index_data: dict) -> float:
        """حساب درجة BM25"""
        stats = index_data["bm25_stats"]
        doc_lengths = stats.get("doc_lengths", {})
        doc_length = doc_lengths.get(doc_id, 500)
        avg_doc_length = stats.get("avg_doc_length", 500)
        doc_freq = stats.get("doc_freq", {})
        idf = stats.get("idf", {})
        
        k1 = 1.5
        b = 0.75
        
        score = 0.0
        for term in set(query_tokens):
            if term not in doc_freq:
                continue
            
            tf = index_data["inverted_index"].get(term, {}).get(doc_id, 0)
            if tf == 0:
                continue
            
            idf_score = idf.get(term, 0)
            
            # صيغة BM25
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
            
            score += idf_score * (numerator / denominator)
        
        return score
    
    def search(self, dataset_name: str, query_tokens: List[str], model_type: str, k: int) -> List[Dict]:
        try:
            index_data = self.load_index(dataset_name)
        except FileNotFoundError:
            return []
        
        doc_ids = index_data["doc_ids"]
        scores = []
        
        for doc_id in doc_ids:
            if model_type == "tfidf":
                score = self.compute_tfidf_score(query_tokens, doc_id, index_data)
            elif model_type == "bm25":
                score = self.compute_bm25_score(query_tokens, doc_id, index_data)
            else:  # hybrid
                tfidf_score = self.compute_tfidf_score(query_tokens, doc_id, index_data)
                bm25_score = self.compute_bm25_score(query_tokens, doc_id, index_data)
                score = 0.5 * tfidf_score + 0.5 * bm25_score
            
            if score > 0:
                scores.append({"doc_id": str(doc_id), "score": score})
        
        # ترتيب تنازلي
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:k]

retrieval = RetrievalService()

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(service="retrieval_service", status="ok")

@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    results = retrieval.search(
        request.dataset_name,
        request.query_tokens,
        request.model_type,
        request.k  # الآن هو int وليس string
    )
    
    return SearchResponse(
        query_id="query_1",
        results=results,
        model_type=request.model_type
    )

if __name__ == "__main__":
    import uvicorn
    from shared.config import SERVICE_PORTS
    
    uvicorn.run(
        "retrieval_service.main:app",
        host="127.0.0.1",
        port=SERVICE_PORTS["retrieval"],
        reload=True,
    )