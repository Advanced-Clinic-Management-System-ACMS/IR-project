"""
Core indexing logic using industry-standard IR libraries:
- sklearn.feature_extraction.text.TfidfVectorizer
- rank_bm25.BM25Okapi
- sentence-transformers + FAISS (offline vector store)
"""
from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from shared.config import DEFAULT_EMBEDDING_MODEL
from shared.schemas import ProcessedDocument


class IndexBuilderCore:
    @staticmethod
    def build_sklearn_tfidf(
        documents: list[ProcessedDocument],
    ) -> tuple[TfidfVectorizer, object, list[str]]:
        corpus = [" ".join(doc.tokens) for doc in documents]
        doc_ids = [doc.doc_id for doc in documents]
        vectorizer = TfidfVectorizer(norm="l2", use_idf=True, sublinear_tf=True)
        matrix = vectorizer.fit_transform(corpus)
        return vectorizer, matrix, doc_ids

    @staticmethod
    def build_rank_bm25(
        documents: list[ProcessedDocument],
    ) -> tuple[BM25Okapi, list[str]]:
        tokenized_corpus = [doc.tokens for doc in documents]
        doc_ids = [doc.doc_id for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        return bm25, doc_ids

    @staticmethod
    def build_sentence_embeddings(
        documents: list[ProcessedDocument],
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 256,
    ) -> tuple[np.ndarray, str]:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        texts = [doc.original_text or " ".join(doc.tokens) for doc in documents]
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(documents) > 1000,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32), model_name

    @staticmethod
    def build_faiss_index(embeddings: np.ndarray):
        import faiss

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        return index

    @staticmethod
    def summary_stats(documents: list[ProcessedDocument]) -> dict:
        total = len(documents)
        avg_len = sum(len(doc.tokens) for doc in documents) / (total or 1)
        return {
            "document_count": total,
            "avg_doc_length": round(avg_len, 2),
            "vocabulary_size": len({token for doc in documents for token in doc.tokens}),
        }
