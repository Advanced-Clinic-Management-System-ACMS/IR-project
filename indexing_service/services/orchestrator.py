"""
This file orchestrates the indexing use case.
It acts as the intermediary between the core mathematical logic and the infrastructure layer,
ensuring strict data flow and execution of the build process.
"""
import json

from shared.config import DEFAULT_EMBEDDING_MODEL, PROCESSED_DIR
from shared.schemas import ProcessedDocument
from indexing_service.core.builder import IndexBuilderCore
from indexing_service.infrastructure.storage import IndexStorageAdapter


class IndexingOrchestrator:
    @staticmethod
    def execute_build(dataset_name: str, documents: list[ProcessedDocument], save_embeddings: bool) -> dict:
        if not documents:
            raise ValueError("No documents provided for indexing.")

        storage = IndexStorageAdapter(dataset_name)

        core_data = IndexBuilderCore.build_core_index(documents)
        vocabulary = core_data["vocabulary"]
        idf = core_data["idf"]

        tf_idf_vectors = IndexBuilderCore.build_tf_idf_vectors(documents, idf, vocabulary)

        storage.save_json("inverted_index.json", core_data["inverted_index"])
        storage.save_json("vocabulary.json", vocabulary)
        storage.save_json("doc_lengths.json", core_data["doc_lengths"])
        storage.save_json("idf.json", idf)
        storage.save_json("bm25_stats.json", core_data["bm25_stats"])
        storage.save_json("tf_idf_vectors.json", tf_idf_vectors)

        embedding_type = "vocabulary_tfidf"
        embedding_model = None
        if save_embeddings:
            embeddings_path = storage.output_dir / "embeddings.npy"
            doc_ids_path = storage.output_dir / "embedding_doc_ids.json"
            current_ids = [doc.doc_id for doc in documents]
            reuse_embeddings = (
                embeddings_path.exists()
                and doc_ids_path.exists()
                and json.loads(doc_ids_path.read_text(encoding="utf-8")) == current_ids
            )
            if reuse_embeddings:
                existing_metadata_path = storage.output_dir / "metadata.json"
                if existing_metadata_path.exists():
                    existing = json.loads(existing_metadata_path.read_text(encoding="utf-8"))
                    embedding_model = existing.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
                else:
                    embedding_model = DEFAULT_EMBEDDING_MODEL
                embedding_type = "sentence_transformer"
            else:
                embeddings, embedding_model = IndexBuilderCore.build_sentence_embeddings(
                    documents, model_name=DEFAULT_EMBEDDING_MODEL
                )
                embedding_type = "sentence_transformer"
                storage.save_numpy("embeddings.npy", embeddings)
                storage.save_json("embedding_doc_ids.json", current_ids)

        processed_payload = [doc.model_dump() for doc in documents]
        processed_jsonl = PROCESSED_DIR / f"{dataset_name}.jsonl"
        if not processed_jsonl.exists() and len(documents) <= 5000:
            storage.save_processed_documents(processed_payload)

        metadata = {
            "dataset_name": dataset_name,
            "document_count": core_data["total_docs"],
            "vocabulary_size": len(vocabulary),
            "avg_doc_length": core_data["avg_doc_length"],
            "index_path": str(storage.output_dir),
            "embedding_type": embedding_type,
            "embedding_model": embedding_model,
        }
        storage.save_json("metadata.json", metadata)

        return metadata
