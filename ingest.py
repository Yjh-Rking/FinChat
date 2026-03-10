import logging
from qdrant_client import models
from core.models import load_markdown, QdrantStore, SQLiteStore
from core.config import config, EMBED_MODEL
from core import split_text
from core.chunking import tiktoken_len

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    docs = load_markdown(config.data.path)
    logger.info(f"Loaded {len(docs)} docs")

    # SQLite insert (Documents + Chunks)
    sqlite_store = SQLiteStore(config.sqlite.path)
    sqlite_store.init_schema()
    # Qdrant insert (Chunks + Embeddings)
    qdrant_store = QdrantStore(
        url=config.qdrant.url, collection=config.qdrant.collection
    )
    qdrant_store.init_collection(vector_size=1024, distance=models.Distance.COSINE)

    for doc, text in docs:
        # SQLite store documents
        sqlite_store.insert_document(doc)

        # Chunking
        chunks = split_text(text, doc.doc_id, chunk_size=256, chunk_overlap=32)
        min_tokens = 20  # 最少 20 个 token
        valid_chunks = []
        for c in chunks:
            text = c.text.strip()
            if not text:
                continue
            # 过滤纯标点
            if text in "。！？.,:;?！":
                continue
            # 过滤 token 太少
            if tiktoken_len(text) < min_tokens:
                continue
            valid_chunks.append(c)
        chunks = valid_chunks

        # QDdrant store chunks + embeddings
        texts = [c.text for c in chunks]
        embeddings = [EMBED_MODEL(text) for text in texts]
        qdrant_store.upsert_chunks(chunks, embeddings)
        logger.info(f"Split into {len(chunks)} chunks")
