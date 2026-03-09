import logging
from typing import List, Optional, Any
from models import RetrievalDoc

logger = logging.getLogger(__name__)


def normalize_scores(docs: List[RetrievalDoc]) -> List[RetrievalDoc]:
    """对分数进行 min-max 归一化"""
    if not docs:
        return docs

    scores = [d.score for d in docs]
    min_score = min(scores)
    max_score = max(scores)

    if max_score - min_score == 0:
        return docs

    for doc in docs:
        doc.score = (doc.score - min_score) / (max_score - min_score)

    return docs


def hybrid_retriever(
    query: str,
    embedding: List[float],
    retrievers: List,
    topk: int = 10,
    dedup: bool = True,
    weights: Optional[List[float]] = None,
    qdrant_store: Optional[Any] = None,
) -> List[RetrievalDoc]:
    """
    灵活混合检索器：支持任意组合的检索器。

    Args:
        query: 用户查询文本
        embedding: 查询的向量表示
        retrievers: 检索器列表，支持以下类型:
            - VectorRetriever: 调用 qdrant_retrieve 方法，需要 embedding 和 qdrant_store
            - BM25Retriever / TavilyWebRetriever: 调用 retrieve 方法，需要 query
        topk: 最终返回的文档数量
        dedup: 是否根据 chunk_id 去重
        weights: 各检索器的权重列表，默认等权重
        qdrant_store: QdrantStore 实例，用于 VectorRetriever

    Returns:
        融合后的 RetrievalDoc 列表
    """
    all_docs = []

    for retriever in retrievers:
        try:
            # 根据检索器类型调用不同的方法
            if hasattr(retriever, "qdrant_retrieve"):
                # VectorRetriever: 需要 qdrant store 和 embedding
                if qdrant_store is None:
                    logger.warning("VectorRetriever 需要 qdrant_store 参数")
                    continue
                docs = retriever.qdrant_retrieve(qdrant_store, embedding, topk=topk)
            elif hasattr(retriever, "retrieve"):
                # BM25Retriever / TavilyWebRetriever: 需要 query
                docs = retriever.retrieve(query, topk=topk)
            else:
                logger.warning(f"未知检索器类型: {type(retriever)}")
                continue

            # 归一化每个检索器的分数
            docs = normalize_scores(docs)
            all_docs.extend(docs)

        except Exception as e:
            logger.error(f"检索器 {type(retriever).__name__} 执行失败: {e}")
            continue

    if not all_docs:
        return []

    # 去重：根据 chunk_id 保留分数最高的文档
    if dedup:
        doc_dict = {}
        for doc in all_docs:
            key = (doc.chunk_id, doc.doc_id)
            if key not in doc_dict or doc.score > doc_dict[key].score:
                doc_dict[key] = doc
        all_docs = list(doc_dict.values())

    # 应用权重（如果提供）
    if weights and len(weights) == len(retrievers):
        for i, retriever in enumerate(retrievers):
            weight = weights[i]
            retriever_name = type(retriever).__name__.lower().replace("retriever", "")
            for doc in all_docs:
                if retriever_name in doc.source.lower():
                    doc.score *= weight

    # 按分数降序排序
    all_docs.sort(key=lambda x: x.score, reverse=True)

    return all_docs[:topk]
