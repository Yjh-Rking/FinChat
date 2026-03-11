"""
RAG 评估脚本

评估 main.py 中 7 种模式的 RAG 性能，使用 ragas 指标
"""

import json
import logging
from typing import List, Dict, Any
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
from core.rag import rag_pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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


def load_test_dataset(path: str = "data/test_dataset.json") -> List[Dict[str, Any]]:
    """加载测试数据集"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_ragas_llm():
    """创建 RAGAS 评估用的 LLM"""
    return llm_factory(
        config.openai.chat,
        client=OpenAI(api_key=config.openai.chat_api, base_url=config.openai.chat_url),
        extra_body={
            "reasoning_split": True,
            "response_format": {"type": "json_object"},
            "raw_json": True,
        },
    )


def get_ragas_embeddings():
    """创建 RAGAS 评估用的 Embeddings"""
    return MyEmbeddings()


def run_single_mode(
    data: List[Dict[str, Any]], mode_name: str, **rag_kwargs
) -> Dict[str, Any]:
    """
    运行单个模式的评估

    Args:
        data: 测试数据集
        mode_name: 模式名称
        **rag_kwargs: 传递给 rag_pipeline 的参数

    Returns:
        评估结果
    """
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    for item in data:
        question = item["question"]
        ground_truth = item["ground_truth"]

        logger.info(f"[{mode_name}] 处理问题: {question}")

        # 调用 rag_pipeline 获取答案和上下文
        result = rag_pipeline(question, return_contexts=True, topk=5, **rag_kwargs)

        if isinstance(result, tuple):
            answer, contexts = result
        else:
            answer = result
            contexts = []

        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(ground_truth)

        logger.info(f"[{mode_name}] 答案: {answer[:100]}...")

    # 构建 RAGAS 数据集
    ragas_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(ragas_data)

    # 执行评估
    llm = get_ragas_llm()
    embeddings = get_ragas_embeddings()

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,  # type: ignore
        embeddings=embeddings,
    )

    return {
        "mode": mode_name,
        "scores": result.scores.to_dict(records=True)[0]  # type: ignore
        if len(result.scores) > 0  # type: ignore
        else {},  # type: ignore
        "mean_scores": {
            "faithfulness": result.scores["faithfulness"].mean(),  # type: ignore
            "answer_relevancy": result.scores["answer_relevancy"].mean(),  # type: ignore
            "context_precision": result.scores["context_precision"].mean(),  # type: ignore
            "context_recall": result.scores["context_recall"].mean(),  # type: ignore
        },
    }


def main():
    """主函数：评估 7 种模式"""
    # 加载测试数据
    test_data = load_test_dataset()
    logger.info(f"加载了 {len(test_data)} 条测试数据")

    # 定义 7 种评估模式
    modes = [
        {
            "name": "1. base (without web)",
            "kwargs": {
                "modes": {"base"},
                "use_web": False,
                "use_rerank": "transformers",
            },
        },
        {
            "name": "2. base (with web)",
            "kwargs": {
                "modes": {"base"},
                "use_web": True,
                "use_rerank": "transformers",
            },
        },
        {
            "name": "3. base + mqe",
            "kwargs": {"modes": {"base", "mqe"}, "use_web": True, "mqe_n": 3},
        },
        {
            "name": "4. base + hyde (完整)",
            "kwargs": {"modes": {"base", "hyde"}, "use_web": True, "hyde_n": 3},
        },
        {
            "name": "5. base + hyde (仅 hypo_doc)",
            "kwargs": {
                "modes": {"base", "hyde"},
                "use_web": True,
                "use_hypo_doc": True,
                "use_hyde_rewrite": False,
            },
        },
        {
            "name": "6. base + hyde (仅 hyde_rewrite)",
            "kwargs": {
                "modes": {"base", "hyde"},
                "use_web": True,
                "use_hypo_doc": False,
                "use_hyde_rewrite": True,
            },
        },
        {
            "name": "7. base + mqe + hyde (完整)",
            "kwargs": {
                "modes": {"base", "mqe", "hyde"},
                "use_web": True,
                "mqe_n": 3,
                "hyde_n": 3,
            },
        },
    ]

    results = []

    # 依次评估每种模式
    for mode in modes:
        logger.info("=" * 50 + f" {mode['name']} " + "=" * 50)
        result = run_single_mode(test_data, mode["name"], **mode["kwargs"])
        results.append(result)
        logger.info(f"[{mode['name']}] 评估完成")

    # 打印汇总结果
    logger.info("\n" + "=" * 80)
    logger.info("评估结果汇总")
    logger.info("=" * 80)

    print(
        "\n{:<35} {:>12} {:>12} {:>12} {:>12}".format(
            "Mode", "Faithfulness", "AnswerRelev", "ContextPrec", "ContextRec"
        )
    )
    print("-" * 80)

    for r in results:
        ms = r["mean_scores"]
        print(
            "{:<35} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}".format(
                r["mode"],
                ms["faithfulness"],
                ms["answer_relevancy"],
                ms["context_precision"],
                ms["context_recall"],
            )
        )

    # 保存详细结果到文件
    output_path = "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"\n详细结果已保存到 {output_path}")


if __name__ == "__main__":
    main()
