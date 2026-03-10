"""
HyDE (Hypothetical Document Embeddings) 查询增强

通过让 LLM 生成"假设文档"来改进检索效果
"""

from typing import List
from core.config import CHAT_MODEL
from core.query.utils import clean_queries


def generate_hypo_docs(query: str, max_chars: int = 200) -> str:
    """
    生成 HyDE 假设回答

    Args:
        query: 用户问题
        max_chars: 假设回答的最大字符数
    Returns:
        假设回答列表
    """

    system_prompt = "你是中文知识库助手。请根据用户问题生成的参考答案文本，不要编造数字，保留原问题中的实体、时间、约束"

    user_prompt = """
        原问题：{question}
        请围绕原问题写一段"可能的参考答案文本"，要求：
        1) 全部简体中文
        2) {max_chars}字以内
        3) 保留原问题中的实体、时间、约束、关键术语、同义表达
        4) 不要编造数据，不需要输出思考
    """

    text = CHAT_MODEL(
        prompt=user_prompt.format(query=query, max_chars=max_chars),
        system_message=system_prompt,
    )

    return text[:max_chars].strip()


def hyde_rewrite(query: str, n: int = 5, original_query: bool = True) -> List[str]:
    """
    HyDE 查询改写

    Args:
        query: 用户问题
        n: 生成查询数量
        original_query: 是否包含原始查询

    Returns:
        增强后的查询列表
    """
    system_prompt = "你是中文检索查询改写助手。请严格输出多条用于向量检索的查询语句。"

    user_prompt = """
        参考文档：{hypo_doc}
        请基于以上参考文档，生成 {n} 条不同的中文检索查询，要求：
        1) 全部简体中文
        2) 每行一条，每条查询都要有意义
        3) 不要编号，不要解释，不要重复
        4) 从参考文档中提取关键信息进行查询
    """

    # Step 1: 生成假设文档
    hypo_doc = generate_hypo_docs(query)

    # Step 2: 基于假设文档生成多个查询
    text = CHAT_MODEL(
        prompt=user_prompt.format(hypo_doc=hypo_doc, n=n),
        system_message=system_prompt,
    )
    queries = clean_queries(text, n)

    if original_query and query not in queries:
        return [query] + queries
    return queries
