"""
SQLite-backed document repository for online top-K text retrieval.
"""
from __future__ import annotations

from shared.document_db import DocumentDatabase


class DocumentRepository:
    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name

    def get_documents_by_ids(self, doc_ids: list[str]) -> dict[str, dict]:
        return DocumentDatabase.get_documents_by_ids(self.dataset_name, doc_ids)

    def get_document(self, doc_id: str) -> dict | None:
        docs = self.get_documents_by_ids([doc_id])
        return docs.get(doc_id)
