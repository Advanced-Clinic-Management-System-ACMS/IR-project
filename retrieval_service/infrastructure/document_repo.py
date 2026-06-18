"""
Infrastructure adapter for reading original document text from local files.
Supports JSON arrays and JSONL for large LoTTE corpora.
"""
from __future__ import annotations

import json

from shared.config import PROCESSED_DIR, RAW_DIR


class DocumentRepository:
    _cache: dict[str, dict[str, dict]] = {}

    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name

    def _load_from_json(self, path) -> dict[str, dict]:
        documents: dict[str, dict] = {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload:
            documents[str(item["doc_id"])] = {
                "doc_id": str(item["doc_id"]),
                "text": item.get("text", item.get("original_text", "")),
                "title": item.get("title"),
            }
        return documents

    def _scan_jsonl_for_ids(self, path, needed: set[str]) -> dict[str, dict]:
        found: dict[str, dict] = {}
        if not needed:
            return found
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                doc_id = str(item["doc_id"])
                if doc_id not in needed:
                    continue
                found[doc_id] = {
                    "doc_id": doc_id,
                    "text": item.get("text", item.get("original_text", "")),
                    "title": item.get("title"),
                }
                if len(found) == len(needed):
                    break
        return found

    def _load_documents(self) -> dict[str, dict]:
        if self.dataset_name in self._cache:
            return self._cache[self.dataset_name]

        documents: dict[str, dict] = {}

        raw_json = RAW_DIR / f"{self.dataset_name}.json"
        if raw_json.exists():
            documents.update(self._load_from_json(raw_json))

        raw_jsonl = RAW_DIR / f"{self.dataset_name}.jsonl"
        if raw_jsonl.exists() and not documents:
            documents.update(self._scan_jsonl_for_ids(raw_jsonl, set()))

        processed_json = PROCESSED_DIR / f"{self.dataset_name}.json"
        if processed_json.exists():
            for doc_id, item in self._load_from_json(processed_json).items():
                documents.setdefault(doc_id, item)

        self._cache[self.dataset_name] = documents
        return documents

    def get_documents_by_ids(self, doc_ids: list[str]) -> dict[str, dict]:
        needed = set(doc_ids)
        found = {doc_id: doc for doc_id, doc in self._load_documents().items() if doc_id in needed}

        missing = needed - set(found)
        if not missing:
            return found

        for path in (
            RAW_DIR / f"{self.dataset_name}.jsonl",
            PROCESSED_DIR / f"{self.dataset_name}.jsonl",
        ):
            if not path.exists():
                continue
            found.update(self._scan_jsonl_for_ids(path, missing - set(found)))
            missing = needed - set(found)
            if not missing:
                break

        return found

    def get_document(self, doc_id: str) -> dict | None:
        docs = self.get_documents_by_ids([doc_id])
        return docs.get(doc_id)
