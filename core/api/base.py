import requests
from openai import OpenAI
from config import config


def rerank_model(query: str, documents: list[str]) -> list[float]:
    """
    使用本地部署的 infinity 服务进行 rerank

    Args:
        query: 用户查询
        documents: 待排序的文档列表
        model: 模型名称，默认使用 config.openai.rerank

    Returns:
        每个文档的相关性分数列表
    """
    if not documents:
        return []

    rerank_url = config.openai.rerank_url

    if not rerank_url:
        raise ValueError("rerank_url is not configured")

    url = f"{rerank_url}/rerank"
    payload = {
        "query": query,
        "documents": documents,
        "model": config.openai.rerank,
    }

    response = requests.post(
        url, json=payload, headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()

    result = response.json()
    print(result)
    # 返回按原始顺序排列的分数
    scores = [0.0] * len(documents)
    for item in result.get("results", []):
        idx = item.get("index", 0)
        score = item.get("score", 0.0)
        scores[idx] = score

    return scores


def embedding_model(text: str) -> list[float]:
    """Create embedding using OpenAI's embeddings API."""
    client = OpenAI(
        api_key=config.openai.embed_api,
        base_url=config.openai.embed_url,
    )
    response = client.embeddings.create(
        model=config.openai.embed,
        input=text,
    )
    return response.data[0].embedding


def chat_model(
    prompt: str, system_message: str = "", extra_body: dict = {"reasoning_split": True}
) -> str:
    """Chat using OpenAI's chat completions API."""
    client = OpenAI(
        api_key=config.openai.chat_api,
        base_url=config.openai.chat_url,
    )
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=config.openai.chat,
        messages=messages,
        temperature=0,
        extra_body=extra_body,
    )
    return response.choices[0].message.content  # type: ignore
