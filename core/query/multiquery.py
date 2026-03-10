"""
Multi-Query 查询增强

通过让 LLM 生成多个查询变体来改进检索召回
"""

import re
from typing import List
from core.config import CHAT_MODEL


def clean_queries(text: str, original_question: str, n: int) -> List[str]:
    """
    清洗 LLM 生成的查询，移除编号等干扰内容

    Args:
        text: LLM 生成的原始文本
        original_question: 原始问题（兜底用）
        n: 期望返回的查询数量

    Returns:
        清洗后的查询列表
    """
    lines = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^[\-\d\.\)\s]+", "", line)  # 去掉可能的编号
        if line:
            lines.append(line)

    # 兜底：把原问题放进去，避免改写跑偏
    if original_question not in lines:
        lines = [original_question] + lines

    # 去重并截断
    dedup = []
    seen = set()
    for q in lines:
        if q not in seen:
            seen.add(q)
            dedup.append(q)

    return dedup[: max(n, 1)]


def generate_multi_queries(question: str, n: int = 5) -> List[str]:
    """
    生成多个查询变体

    Args:
        question: 原始问题
        n: 生成数量

    Returns:
        查询列表
    """
    system_prompt = "你是中文检索查询改写助手。请严格输出多条用于向量检索的查询语句。"

    user_prompt = """
        原问题：{question}
        请把原问题改写为 {n} 条不同角度的中文检索查询，要求：
        1) 全部简体中文
        2) 每行一条，不要编号
        3) 不要解释，不要多余文本
        4) 保留原问题中的实体、时间、约束
    """

    text = CHAT_MODEL(
        prompt=user_prompt.format(n=n, question=question),
        system_message=system_prompt,
    )

    return clean_queries(text, question, n)


def multiquery_enhance(
    question: str, n: int = 5, include_original: bool = True
) -> List[str]:
    """
    Multi-Query 查询增强

    Args:
        question: 用户问题
        n: 生成查询数量
        include_original: 是否包含原始查询

    Returns:
        增强后的查询列表
    """
    queries = generate_multi_queries(question, n=n)

    if include_original and question not in queries:
        return [question] + queries
    return queries
