"""
This file manages the connection to MongoDB. 
The retrieval service strictly owns this connection to fetch document snippets.
"""
from typing import Any
from pymongo import MongoClient
from pymongo.collection import Collection
from shared.config import MONGO_URI, MONGO_DB_NAME, MONGO_DOCS_COLLECTION

class MongoDocumentRepository:
    def __init__(self) -> None:
        self.client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        self.collection: Collection = self.client[MONGO_DB_NAME][MONGO_DOCS_COLLECTION]

    def get_documents_by_ids(self, doc_ids: list[str]) -> dict[str, dict[str, Any]]:
        cursor = self.collection.find({"doc_id": {"$in": doc_ids}}, {"_id": 0})
        return {doc["doc_id"]: doc for doc in cursor}