import logging
from qdrant_client import models
from models import load_markdown, QdrantStore, SQLiteStore
from utils import config, split_text, query_embedding

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    docs = load_markdown(config.data.path)
    logger.info(f"Loaded {len(docs)} docs")

    # SQLite insert (Documents + Chunks)
    sqlite_store = SQLiteStore(config.sqlite.path)
    sqlite_store.init_schema()
    for doc, text in docs:
        sqlite_store.insert_document(doc)
        chunks = split_text(text, doc.doc_id, chunk_size=256, chunk_overlap=32)
        sqlite_store.insert_chunks(chunks)
        logger.info(f"Split into {len(chunks)} chunks")

    # Qdrant insert (Chunks + Embeddings)
    qdrant_store = QdrantStore(
        host=config.qdrant.host,
        port=config.qdrant.port,
        collection=config.qdrant.collection_name,
    )
    qdrant_store.init_collection(vector_size=1024, distance=models.Distance.COSINE)
    texts = [c.text for c in chunks]
    embeddings = [query_embedding(text) for text in texts]
    qdrant_store.upsert_chunks(chunks, embeddings)
