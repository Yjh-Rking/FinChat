"""
HyDE (Hypothetical Document Embeddings) 查询增强

通过让 LLM 生成多个"假设文档"来改进检索效果
"""

from typing import List
from core.config import CHAT_MODEL


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


def hyde_enhance(query: str, include_original: bool = True) -> List[str]:
    """
    HyDE 查询增强

    Args:
        query: 用户问题
        n: 生成假设回答数量
        include_original: 是否包含原始查询

    Returns:
        增强后的查询列表
    """
    hypo_doc = generate_hypo_docs(query)

    if include_original:
        return [query + hypo_doc]
    return [hypo_doc]
