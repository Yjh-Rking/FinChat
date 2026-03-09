import logging
from utils.config import config, CHAT_MODEL
from utils.retriever import hybrid_retriever
from utils.rerank import llm_cross_encoder_rerank

logger = logging.getLogger(__name__)


def ingest():
    import uuid
    import sqlite3
    from qdrant_client import QdrantClient
    from utils.config import EMBED_MODEL
    from utils.chunking import load_markdown, split_docs
    from utils.qdrant import get_vector_store
    from utils.sqlite import init_table

    docs = load_markdown(config.data.path)
    chunks = split_docs(256, 32, docs)
    logger.info(f"Loaded {len(docs)} docs, split into {len(chunks)} chunks.")

    init_table(config.sqlite.path)
    with sqlite3.connect(config.sqlite.path) as conn:
        ids = []
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            text = chunk.page_content
            source = chunk.metadata.get("source", "")
            conn.execute(
                """
                INSERT INTO chunks (chunk_id, doc_id, text)
                VALUES (?, ?, ?)
                """,
                (chunk_id, source, text),
            )
            ids.append(chunk_id)
        conn.commit()
    client = QdrantClient(host=config.qdrant.host, port=config.qdrant.port)
    vector_store = get_vector_store(
        client, config.qdrant.collection_name, 1024, EMBED_MODEL
    )
    vector_store.add_documents(documents=chunks, ids=ids, batch_size=32)
    logger.info("Ingest done:", len(chunks))
    client.close()
    # results = vector_store.similarity_search("what is ally?", k=1)
    # logger.info(results)


if __name__ == "__main__":
    # 数据预处理
    # ingest()

    query = "哪个分析师做了顺丰的研报？"
    docs = hybrid_retriever(query, 10)
    logging.info(f"检索到 {len(docs)} 条相关文档，准备重排序...")
    docs = llm_cross_encoder_rerank(query, docs, 5)
    logging.info(f"重排序完成，最终选取 {len(docs)} 条文档作为回答依据。")

    context = "\n\n".join(d.page_content for d in docs)
    prompt = f"""
    请根据以下内容回答问题
    {context}

    问题:
    {query}
    """
    ans = CHAT_MODEL.invoke(prompt).content
    logger.info("\n回答:\n", ans)
