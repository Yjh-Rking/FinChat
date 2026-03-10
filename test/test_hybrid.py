"""测试 hybrid_retriever 函数"""

import sys

sys.path.insert(0, ".")

from core.retriever import (
    VectorRetriever,
    BM25Retriever,
    TavilyWebRetriever,
    hybrid_retriever,
)
from core.config import config, EMBED_MODEL
from core.models import SQLiteStore, QdrantStore


def test_hybrid_retriever():
    """测试混合检索器"""
    query = "顺丰的研报分析师是谁？"
    embedding = EMBED_MODEL(query)

    # 初始化 SQLite store
    sqlite_store = SQLiteStore(config.sqlite.path)

    # 初始化 Qdrant store
    qdrant_store = QdrantStore(
        url=config.qdrant.url, collection=config.qdrant.collection
    )

    # Vector Retriever
    vector_retriever = VectorRetriever(sqlite_store)

    # BM25 Retriever - 需要先获取所有 chunks
    all_chunks = [
        sqlite_store.get_chunk_by_id(row["chunk_id"])
        for row in sqlite_store.conn.execute("SELECT chunk_id FROM chunks")
    ]
    all_chunks = [c for c in all_chunks if c is not None]
    bm25_retriever = BM25Retriever(all_chunks, language="zh")

    # Web Retriever (如果环境变量存在)
    retrievers = [vector_retriever, bm25_retriever]
    web_retriever = TavilyWebRetriever(api_key=config.tavily.token)
    retrievers.append(web_retriever)
    weights = [0.4, 0.3, 0.3]

    # 混合检索
    results = hybrid_retriever(
        query=query,
        embedding=embedding,
        retrievers=retrievers,
        qdrant_store=qdrant_store,
        topk=5,
        weights=weights,
    )

    # 打印结果
    print(f"\n查询: {query}")
    print(f"返回 {len(results)} 条结果:\n")
    for i, doc in enumerate(results, 1):
        print(f"--- 结果 {i} ---")
        print(f"来源: {doc.source}")
        print(f"分数: {doc.score:.4f}")
        print(f"内容: {doc.text[:200]}...")
        print()

    return results


if __name__ == "__main__":
    test_hybrid_retriever()
