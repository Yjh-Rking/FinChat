import re
from typing import List


def clean_queries(text: str, n: int) -> List[str]:
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

    dedup = []
    for q in lines:
        dedup.append(q)

    return dedup[: max(n, 1)]
