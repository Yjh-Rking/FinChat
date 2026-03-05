import re
from utils.config import CHAT_MODEL
from langchain_core.messages import HumanMessage, SystemMessage


def extract_score(text, default=0.0):
    """
    从大模型输出中提取 0-10 分的数字
    """
    # 匹配最后出现的 0-10 数字（整数或浮点数）
    matches = re.findall(r"\b([0-9]|10)(?:\.\d+)?\b", text)
    if matches:
        # 返回最后一个匹配的数字
        return float(matches[-1])
    return default


def llm_cross_encoder_rerank(query, docs, topk):
    scores = []
    for doc in docs:
        resp = CHAT_MODEL.invoke(
            [
                SystemMessage(
                    content=(
                        "你是一个相关性评分专家。你的任务是："
                        "给定一个问题和一个文档，请输出它们的相关性分数（0-10分，可以带小数）。"
                        "【重要】你只能输出一个数字，不要任何解释、推理、标点、单位或额外字符。"
                        "例如：8.5"
                    )
                ),
                HumanMessage(content=f"问题：{query}\n\n文档: {doc}"),
            ]
        )
        try:
            score = extract_score(resp.content)
        except ValueError:
            print(f"无法解析分数: '{resp.content}'")
            score = 0.0
        scores.append(score)

    # 按分数从高到低排序
    sorted_docs_scores = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    # 返回 topk
    return sorted_docs_scores[:topk]
