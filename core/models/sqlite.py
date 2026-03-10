import json
import sqlite3
from typing import Optional
from .types import Document, Chunk

CREATE_DOCUMENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT,
    source TEXT,
    metadata JSON
);
"""

CREATE_CHUNKS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT,
    text TEXT,
    metadata JSON,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);
"""

CREATE_INDEX_CHUNKS_DOC_ID_SQL = """
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
"""


class SQLiteStore:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def init_schema(self):
        self.conn.execute(CREATE_DOCUMENTS_TABLE_SQL)
        self.conn.execute(CREATE_CHUNKS_TABLE_SQL)
        self.conn.execute(CREATE_INDEX_CHUNKS_DOC_ID_SQL)
        self.conn.commit()

    def insert_document(self, doc: Document):
        self.conn.execute(
            """
            INSERT OR REPLACE INTO documents
            VALUES (?, ?, ?, ?)
            """,
            (
                doc.doc_id,
                doc.title,
                doc.source,
                json.dumps(doc.metadata),
            ),
        )
        self.conn.commit()

    def insert_chunks(self, chunks: list[Chunk]):
        rows = [
            (
                c.chunk_id,
                c.doc_id,
                c.text,
                json.dumps(c.metadata),
            )
            for c in chunks
        ]
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO chunks
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def get_all_chunks(self, limit: Optional[int] = None) -> list[Chunk]:
        query = "SELECT * FROM chunks"
        if limit:
            query += f" LIMIT {limit}"
        rows = self.conn.execute(query).fetchall()
        chunks = []
        for row in rows:
            chunks.append(
                Chunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    text=row["text"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                )
            )
        return chunks

    def get_random_chunks(self, k: int = 10) -> list[Chunk]:
        rows = self.conn.execute(
            "SELECT * FROM chunks ORDER BY RANDOM() LIMIT ?", (k,)
        ).fetchall()
        chunks = []
        for row in rows:
            chunks.append(
                Chunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    text=row["text"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                )
            )
        return chunks

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        row = self.conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()

        if row:
            return Chunk(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                text=row["text"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
        return None
