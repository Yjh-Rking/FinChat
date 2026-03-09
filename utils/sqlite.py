import sqlite3
import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def init_table(sqlite_path):
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS chunks(
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            text TEXT
        )""")
        conn.commit()


def load_docs_from_sqlite(sqlite_path, limit=None, doc_id=None):
    docs = []
    with sqlite3.connect(sqlite_path) as conn:
        conn.row_factory = sqlite3.Row  # 允许按列名访问
        sql = "SELECT chunk_id, doc_id, text FROM chunks"
        params = []
        conditions = []

        if doc_id:
            conditions.append("doc_id = ?")
            params.append(doc_id)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        sql += " ORDER BY rowid"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)

        rows = conn.execute(sql, params).fetchall()

        for r in rows:
            docs.append(
                Document(
                    page_content=r["text"] or "",
                    metadata={
                        "chunk_id": r["chunk_id"],
                        "doc_id": r["doc_id"],
                        "source": "sqlite_chunks",
                    },
                )
            )
    return docs
