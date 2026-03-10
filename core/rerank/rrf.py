"""
RRF (Reciprocal Rank Fusion) 重排序

用于融合多个检索结果
"""

from typing import List
from core.models import RetrievalDoc


def reciprocal_rank_fusion(
    ranked_lists: List[List[RetrievalDoc]],
    k: int = 60,
) -> List[RetrievalDoc]:
    """
    RRF (Reciprocal Rank Fusion) 融合多个排序结果

    Args:
        ranked_lists: 多个已排序的文档列表
        k: RRF 参数，默认 60

    Returns:
        融合后的排序结果
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, RetrievalDoc] = {}

    for docs in ranked_lists:
        for rank, doc in enumerate(docs, start=1):
            key = doc.chunk_id
            if key not in doc_map:
                doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    # 按 RRF 分数排序
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    result = []
    for key, score in fused:
        doc = doc_map[key]
        doc.score = score  # 更新 doc 的 score 为 RRF 分数
        result.append(doc)

    return result
