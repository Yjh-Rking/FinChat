import logging
import requests
from typing import List
from .config import config

logger = logging.getLogger(__name__)


def query_embedding(query: str) -> List[float]:
    # Todo: Local model
    # Api call
    try:
        response = requests.post(
            url=config.embed.url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.embed.token}",
            },
            json={"model": config.embed.model, "input": query},
        )
        response.raise_for_status()  # 如果状态码不是 2xx，会抛出 HTTPError
        result = response.json()
        q_embedding = result["data"][0]["embedding"]
        return q_embedding
    except requests.exceptions.RequestException as e:
        # 包括连接错误、超时、HTTPError 等所有 requests 相关异常
        logger.error(f"Request failed: {e}")
        raise
    except (KeyError, IndexError, ValueError) as e:
        # JSON 格式不符合预期
        logger.error(f"Response parsing error: {e}")
        raise
    except Exception as e:
        # 其他未预期异常
        logger.error(f"Unexpected error: {e}")
        raise
