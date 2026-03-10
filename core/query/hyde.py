"""
HyDE (Hypothetical Document Embeddings) 查询增强

通过让 LLM 生成多个"假设文档"来改进检索效果
"""

import re
from typing import List
from core.config import CHAT_MODEL


def clean_hyde_docs(text: str, original_question: str, n: int) -> List[str]:
    """
    清洗 LLM 生成的假设回答，移除编号等干扰内容

    Args:
        text: LLM 生成的原始文本
        original_question: 原始问题
        n: 期望返回的数量

    Returns:
        清洗后的假设回答列表
    """
    lines = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^[\-\d\.\)\s]+", "", line)  # 去掉可能的编号
        if line and len(line) > 10:  # 过滤掉太短的行
            lines.append(line)

    # 去重并截断
    dedup = []
    seen = set()
    for doc in lines:
        if doc not in seen:
            seen.add(doc)
            dedup.append(doc)

    return dedup[: max(n, 1)]


def generate_hypo_docs(question: str, n: int = 3) -> List[str]:
    """
    生成多个 HyDE 假设回答

    Args:
        question: 用户问题
        n: 生成数量，默认 3

    Returns:
        假设回答列表
    """
    system_prompt = (
        "你是中文知识库助手。请根据用户问题生成可能出现在知识库中的参考答案文本。"
    )

    user_prompt = """请围绕这个问题生成 {n} 条不同的"可能的参考答案文本"（用于检索召回）。
    要求：
    1) 全部简体中文
    2) 每行一条，不要编号
    3) 100~200字
    4) 包含关键术语、同义表达
    5) 不要输出思考过程或标签

    问题：{question}"""

    text = CHAT_MODEL(
        prompt=user_prompt.format(n=n, question=question),
        system_message=system_prompt,
    )

    return clean_hyde_docs(text, question, n)


def hyde_enhance(question: str, n: int = 3, include_original: bool = True) -> List[str]:
    """
    HyDE 查询增强

    Args:
        question: 用户问题
        n: 生成假设回答数量
        include_original: 是否包含原始查询

    Returns:
        增强后的查询列表
    """
    hypo_docs = generate_hypo_docs(question, n=n)

    if include_original:
        return [question] + hypo_docs
    return hypo_docs
