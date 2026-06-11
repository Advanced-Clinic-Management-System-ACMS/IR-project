from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection

from shared.config import (
    MONGO_DB_NAME,
    MONGO_DOCS_COLLECTION,
    MONGO_URI,
)


class DocumentStore:
    """Stores raw documents in MongoDB for fast online retrieval by doc_id."""

    def __init__(
        self,
        uri: str = MONGO_URI,
        db_name: str = MONGO_DB_NAME,
        collection_name: str = MONGO_DOCS_COLLECTION,
    ) -> None:
        self.client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        self.collection: Collection = self.client[db_name][collection_name]

    def ping(self) -> bool:
        self.client.admin.command("ping")
        return True

    def insert_documents(self, documents: list[dict[str, Any]]) -> int:
        if not documents:
            return 0
        self.collection.delete_many({})
        result = self.collection.insert_many(documents)
        return len(result.inserted_ids)

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        return self.collection.find_one({"doc_id": doc_id}, {"_id": 0})

    def get_documents_by_ids(self, doc_ids: list[str]) -> list[dict[str, Any]]:
        cursor = self.collection.find({"doc_id": {"$in": doc_ids}}, {"_id": 0})
        docs = {doc["doc_id"]: doc for doc in cursor}
        return [docs[doc_id] for doc_id in doc_ids if doc_id in docs]
