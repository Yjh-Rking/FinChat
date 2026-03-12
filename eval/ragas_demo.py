from typing import List
from openai import OpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import llm_factory
from ragas.embeddings import BaseRagasEmbeddings
from config import config
from core.api import embedding_model


class MyEmbeddings(BaseRagasEmbeddings):
    def embed_documents(self, docs: list[str]) -> list[list[float]]:
        embeddings = []
        for doc in docs:
            embedding = embedding_model(doc)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        return embedding_model(query)

    async def aembed_query(self, text: str) -> List[float]:
        raise NotImplementedError

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


llm = llm_factory(
    config.openai.chat_model,
    client=OpenAI(api_key=config.openai.chat_api, base_url=config.openai.chat_url),
    extra_body={
        "reasoning_split": True,
        "response_format": {"type": "json_object"},
        "raw_json": True,  # 关键，告诉模型返回纯 JSON
    },
)

data = {
    "question": ["公司2024年营业收入是多少？"],
    "answer": ["公司2024年营业收入为120亿元"],
    "contexts": [["公司2024年实现营业收入120亿元，同比增长15%。"]],
    "ground_truth": ["120亿元"],
}
dataset = Dataset.from_dict(data)

precomputed_embeddings = MyEmbeddings()
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=llm,  # type: ignore
    embeddings=precomputed_embeddings,
)

print(result)
