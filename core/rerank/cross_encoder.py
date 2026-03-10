import re
import logging
from typing import List
from sentence_transformers import CrossEncoder
from core.models import RetrievalDoc
from config import config
from core.api import chat_model

logger = logging.getLogger(__name__)
reranker = CrossEncoder(config.rerank.model, device="cpu", max_length=512)


def extract_score(text: str, default: float = 0.0) -> float:
    """
    从大模型输出中提取 0-10 分的数字
    """
    # 匹配最后出现的 0-10 数字（整数或浮点数）
    matches = re.findall(r"\b([0-9]|10)(?:\.\d+)?\b", text)
    if matches:
        # 返回最后一个匹配的数字
        return float(matches[-1])
    return default


def llm_cross_encoder_rerank(
    query: str, docs: List[RetrievalDoc], topk: int = 10
) -> List[RetrievalDoc]:
    if not docs:
        return []

    system_prompt = (
        "你是一个相关性评分专家。你的任务是："
        "给定一个问题和一个文档，请输出它们的相关性分数（0-10分，可以带小数）。"
        "【重要】你只能输出一个数字，不要任何解释、推理、标点、单位或额外字符。"
        "例如：8.5"
    )

    for doc in docs:
        user_prompt = f"问题：{query}\n\n文档: {doc.text}"
        try:
            content = chat_model(user_prompt, system_prompt)
            score = extract_score(content)
        except Exception as e:
            logger.info(f"LLM 调用失败: {e}")
            score = 0.0

        # 将分数归一化到 0-1 范围
        doc.score = score / 10.0

    # 按分数从高到低排序
    sorted_docs = sorted(docs, key=lambda x: x.score, reverse=True)

    return sorted_docs[:topk]


def transformers_cross_encoder_rerank(
    query: str, docs: List[RetrievalDoc], topk: int = 10
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
    scores = reranker.predict(query_doc_pairs, show_progress_bar=False)

    # 将分数赋值给文档
    for doc, score in zip(docs, scores):
        doc.score = float(score)

    # 按分数从高到低排序
    sorted_docs = sorted(docs, key=lambda x: x.score, reverse=True)

    return sorted_docs[:topk]
