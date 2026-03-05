import sqlite3
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from utils.config import config, EMBED_MODEL


def get_vector_retriever(retriever_k=2):
    vectorstore = QdrantVectorStore.from_existing_collection(
        collection_name=config.qdrant.collection_name,
        host=config.qdrant.host,
        port=config.qdrant.port,
        embedding=EMBED_MODEL,
    )
    return vectorstore.as_retriever(search_kwargs={"k": retriever_k})


def get_bm25_retriever(retriever_k=2):
    with sqlite3.connect(config.sqlite.path) as conn:
        rows = conn.execute("SELECT text FROM chunks").fetchall()
        docs = [r[0] for r in rows]
        retriever = BM25Retriever.from_texts(docs, k=retriever_k)
    return retriever


def hybrid_retriever(query, retriever_k=2):
    vector = get_vector_retriever(retriever_k=retriever_k)
    bm25 = get_bm25_retriever(retriever_k=retriever_k)

    docs = vector.invoke(query)
    docs += bm25.invoke(query)

    return docs
