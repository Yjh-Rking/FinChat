import logging
import requests
from typing import List
from core.models import RetrievalDoc

logger = logging.getLogger(__name__)


class TavilyWebRetriever:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com/search"

    def retrieve(self, query: str, topk: int = 10) -> List[RetrievalDoc]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "query": query,
            "max_results": topk,
            "search_depth": "basic",
        }

        try:
            resp = requests.post(self.base_url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        except Exception as e:
            logger.error(f"Tavily 搜索失败: {e}")
            return []

        docs = []
        results = data.get("results", [])
        for i, item in enumerate(results):
            title = item.get("title") or ""
            snippet = item.get("content") or ""
            url = item.get("url") or ""

            # 构造 RetrievalDoc
            docs.append(
                RetrievalDoc(
                    chunk_id=f"web_{i}",
                    doc_id=f"web_{i}",
                    text=f"{title}\n{snippet}",
                    score=float(topk - i) / topk,
                    source="tavily_web",
                    metadata={"url": url, "title": title},
                )
            )

        return docs
