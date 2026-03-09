import re
import logging
from typing import List
from openai import OpenAI
from models import RetrievalDoc
from utils.config import config

logger = logging.getLogger(__name__)

# 初始化 OpenAI 客户端
OPENAI_CLIENT = OpenAI(
    base_url=config.chat.url,
    api_key=config.chat.token,
)


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


def llm_chat(prompt: str, system_message: str = "") -> str:
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    response = OPENAI_CLIENT.chat.completions.create(
        model=config.chat.model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content  # type: ignore


def llm_cross_encoder_rerank(
    query: str, docs: List[RetrievalDoc], topk: int = 5
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
            content = llm_chat(user_prompt, system_prompt)
            score = extract_score(content)
        except Exception as e:
            logger.info(f"LLM 调用失败: {e}")
            score = 0.0

        # 将分数归一化到 0-1 范围
        doc.score = score / 10.0

    # 按分数从高到低排序
    sorted_docs = sorted(docs, key=lambda x: x.score, reverse=True)

    return sorted_docs[:topk]
