"""
Multi-Query 查询增强

通过让 LLM 生成多个查询变体来改进检索召回
"""

from typing import List
from core.api import chat_model
from core.query import clean_queries


def generate_multi_queries(query: str, n: int = 5) -> List[str]:
    """
    生成多个查询变体

    Args:
        query: 原始问题
        n: 生成数量

    Returns:
        查询列表
    """
    system_prompt = "你是中文检索查询改写助手。请严格输出多条用于向量检索的查询语句。"

    user_prompt = """
        原问题：{question}
        请把原问题改写为 {n} 条不同角度的中文检索查询，要求：
        1) 全部简体中文
        2) 每行一条，每条查询都要有意义
        3) 不要编号，不要解释，不要重复，不要无效改写
        4) 保留原问题中的实体、时间、约束
    """

    text = chat_model(
        prompt=user_prompt.format(n=n, question=query),
        system_message=system_prompt,
    )

    return clean_queries(text, n)


def multiquery_enhance(
    query: str, n: int = 5, original_query: bool = True
) -> List[str]:
    """
    Multi-Query 查询增强

    Args:
        query: 用户问题
        n: 生成查询数量
        original_query: 是否包含原始查询

    Returns:
        增强后的查询列表
    """
    queries = generate_multi_queries(query, n=n)

    if original_query and query not in queries:
        return [query] + queries
    return queries
