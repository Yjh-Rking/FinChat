from typing import List
from sentence_transformers import CrossEncoder
from models import RetrievalDoc

reranker = CrossEncoder("BAAI/bge-reranker-base")


def transformers_cross_encoder_rerank(
    query: str, docs: List[RetrievalDoc], topk: int = 5
) -> List[RetrievalDoc]:
    """
    使用 BGE Reranker 模型对文档进行重排序

    Args:
        query: 用户查询
        docs: 待重排序的文档列表
        topk: 返回的文档数量

    Returns:
        重排序后的文档列表
    """
    if not docs:
        return []

    # 构建 [query, doc] 对
    query_doc_pairs = [[query, doc.text] for doc in docs]

    # 获取相关性分数
    scores = reranker.predict(query_doc_pairs)

    # 将分数赋值给文档
    for doc, score in zip(docs, scores):
        doc.score = float(score)

    # 按分数从高到低排序
    sorted_docs = sorted(docs, key=lambda x: x.score, reverse=True)

    return sorted_docs[:topk]
