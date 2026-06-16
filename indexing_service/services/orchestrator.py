"""
This file orchestrates the indexing use case.
It acts as the intermediary between the core mathematical logic and the infrastructure layer,
ensuring strict data flow and execution of the build process.
"""
from shared.schemas import ProcessedDocument
from indexing_service.core.builder import IndexBuilderCore
from indexing_service.infrastructure.storage import IndexStorageAdapter

class IndexingOrchestrator:
    @staticmethod
    def execute_build(dataset_name: str, documents: list[ProcessedDocument], save_embeddings: bool) -> dict:
        if not documents:
            raise ValueError("No documents provided for indexing.")

        # 1. Initialize Infrastructure
        storage = IndexStorageAdapter(dataset_name)

        # 2. Execute Core Business Logic
        core_data = IndexBuilderCore.build_core_index(documents)
        vocabulary = core_data["vocabulary"]
        idf = core_data["idf"]

        tf_idf_vectors = IndexBuilderCore.build_tf_idf_vectors(documents, idf, vocabulary)

        # 3. Execute Infrastructure/Storage Operations
        storage.save_json("inverted_index.json", core_data["inverted_index"])
        storage.save_json("vocabulary.json", vocabulary)
        storage.save_json("doc_lengths.json", core_data["doc_lengths"])
        storage.save_json("idf.json", idf)
        storage.save_json("bm25_stats.json", core_data["bm25_stats"])
        storage.save_json("tf_idf_vectors.json", tf_idf_vectors)

        if save_embeddings:
            embeddings = IndexBuilderCore.build_simple_embeddings(documents, vocabulary)
            storage.save_numpy("embeddings.npy", embeddings)
            storage.save_json("embedding_doc_ids.json", [doc.doc_id for doc in documents])

        # Save the processed raw payload
        processed_payload = [doc.model_dump() for doc in documents]
        storage.save_processed_documents(processed_payload)

        # 4. Generate and return metadata for the controller
        metadata = {
            "dataset_name": dataset_name,
            "document_count": core_data["total_docs"],
            "vocabulary_size": len(vocabulary),
            "avg_doc_length": core_data["avg_doc_length"],
            "index_path": str(storage.output_dir),
        }
        storage.save_json("metadata.json", metadata)
        
        return metadata