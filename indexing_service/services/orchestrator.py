"""
Indexing orchestrator — builds library-based indexes and offline FAISS vector store.
"""
from __future__ import annotations

import json

from shared.config import DEFAULT_EMBEDDING_MODEL, PROCESSED_DIR
from shared.document_db import DocumentDatabase
from shared.schemas import ProcessedDocument
from indexing_service.core.builder import IndexBuilderCore
from indexing_service.infrastructure.storage import IndexStorageAdapter


class IndexingOrchestrator:
    @staticmethod
    def execute_build(dataset_name: str, documents: list[ProcessedDocument], save_embeddings: bool) -> dict:
        if not documents:
            raise ValueError("No documents provided for indexing.")

        storage = IndexStorageAdapter(dataset_name)
        stats = IndexBuilderCore.summary_stats(documents)

        vectorizer, tfidf_matrix, tfidf_doc_ids = IndexBuilderCore.build_sklearn_tfidf(documents)
        bm25_model, bm25_doc_ids = IndexBuilderCore.build_rank_bm25(documents)

        storage.save_joblib("tfidf_vectorizer.joblib", vectorizer)
        storage.save_sparse_matrix("tfidf_matrix.npz", tfidf_matrix)
        storage.save_json_gz("tfidf_doc_ids.json.gz", tfidf_doc_ids)

        storage.save_joblib("bm25_model.joblib", bm25_model)
        storage.save_json_gz("bm25_doc_ids.json.gz", bm25_doc_ids)

        embedding_type = None
        embedding_model = None
        current_ids = [doc.doc_id for doc in documents]

        if save_embeddings:
            embeddings_npz = storage.output_dir / "embeddings.npz"
            doc_ids_gz = storage.output_dir / "embedding_doc_ids.json.gz"
            faiss_path = storage.output_dir / "faiss.index"

            reuse = False
            if embeddings_npz.exists() and doc_ids_gz.exists() and faiss_path.exists():
                with __import__("gzip").open(doc_ids_gz, "rt", encoding="utf-8") as handle:
                    existing_ids = json.load(handle)
                reuse = existing_ids == current_ids

            if reuse:
                embedding_type = "sentence_transformer"
                meta_path = storage.output_dir / "metadata.json"
                if meta_path.exists():
                    embedding_model = json.loads(meta_path.read_text(encoding="utf-8")).get(
                        "embedding_model", DEFAULT_EMBEDDING_MODEL
                    )
                else:
                    embedding_model = DEFAULT_EMBEDDING_MODEL
            else:
                embeddings, embedding_model = IndexBuilderCore.build_sentence_embeddings(
                    documents, model_name=DEFAULT_EMBEDDING_MODEL
                )
                faiss_index = IndexBuilderCore.build_faiss_index(embeddings)
                storage.save_embeddings_compressed("embeddings.npz", embeddings)
                storage.save_json_gz("embedding_doc_ids.json.gz", current_ids)
                storage.save_faiss_index(faiss_index)
                embedding_type = "sentence_transformer"

        DocumentDatabase.upsert_from_processed(
            dataset_name,
            [doc.model_dump() for doc in documents],
        )

        processed_jsonl = PROCESSED_DIR / f"{dataset_name}.jsonl"
        if not processed_jsonl.exists() and len(documents) <= 5000:
            storage.save_processed_documents([doc.model_dump() for doc in documents])

        metadata = {
            "dataset_name": dataset_name,
            "document_count": stats["document_count"],
            "vocabulary_size": stats["vocabulary_size"],
            "avg_doc_length": stats["avg_doc_length"],
            "index_path": str(storage.output_dir),
            "index_format": "library_v2",
            "tfidf_library": "sklearn.TfidfVectorizer",
            "bm25_library": "rank_bm25.BM25Okapi",
            "embedding_type": embedding_type,
            "embedding_model": embedding_model,
            "vector_store": "faiss.IndexFlatIP" if save_embeddings else None,
            "storage": "compressed_joblib_npz_gzip",
            "documents_db": "sqlite:data/documents.db",
        }
        storage.save_json("metadata.json", metadata)
        return metadata
