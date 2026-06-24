"""
SQLite storage for original document text (examiner requirement).
Online retrieval fetches top-K document content from this database.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from shared.config import DATA_DIR
from shared.schemas import DocumentInput

DB_PATH = DATA_DIR / "documents.db"


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            dataset_name TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            text TEXT NOT NULL,
            title TEXT,
            PRIMARY KEY (dataset_name, doc_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_dataset ON documents(dataset_name)"
    )
    conn.commit()
    return conn


class DocumentDatabase:
    @staticmethod
    def upsert_documents(dataset_name: str, documents: list[DocumentInput]) -> int:
        if not documents:
            return 0
        conn = _connect()
        conn.executemany(
            """
            INSERT INTO documents (dataset_name, doc_id, text, title)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(dataset_name, doc_id) DO UPDATE SET
                text = excluded.text,
                title = excluded.title
            """,
            [
                (dataset_name, str(doc.doc_id), doc.text, doc.title)
                for doc in documents
            ],
        )
        conn.commit()
        count = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE dataset_name = ?",
            (dataset_name,),
        ).fetchone()[0]
        conn.close()
        return count

    @staticmethod
    def upsert_from_processed(
        dataset_name: str,
        processed: list[dict],
    ) -> int:
        documents = [
            DocumentInput(
                doc_id=str(item["doc_id"]),
                text=item.get("original_text", ""),
                title=item.get("title"),
            )
            for item in processed
        ]
        return DocumentDatabase.upsert_documents(dataset_name, documents)

    @staticmethod
    def get_documents_by_ids(dataset_name: str, doc_ids: list[str]) -> dict[str, dict]:
        if not doc_ids:
            return {}
        conn = _connect()
        placeholders = ",".join("?" for _ in doc_ids)
        rows = conn.execute(
            f"""
            SELECT doc_id, text, title
            FROM documents
            WHERE dataset_name = ? AND doc_id IN ({placeholders})
            """,
            [dataset_name, *doc_ids],
        ).fetchall()
        conn.close()
        return {
            str(doc_id): {
                "doc_id": str(doc_id),
                "text": text or "",
                "title": title,
            }
            for doc_id, text, title in rows
        }

    @staticmethod
    def count(dataset_name: str) -> int:
        conn = _connect()
        row = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE dataset_name = ?",
            (dataset_name,),
        ).fetchone()
        conn.close()
        return int(row[0]) if row else 0
