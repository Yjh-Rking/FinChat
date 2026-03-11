"""
RAG 评估脚本

评估 main.py 中7种情况下的分数，使用 RAGAS 指标
"""

import json
import logging
from typing import List, Dict, Any

from openai import OpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import llm_factory
from ragas.embeddings import BaseRagasEmbeddings

from config import config
from core.api import embedding_model
from core.rag import rag_pipeline

# 初始化 RAGAS 指标
faithfulness = Faithfulness()
answer_relevancy = AnswerRelevancy()
context_precision = ContextPrecision()
context_recall = ContextRecall()

logger = logging.getLogger(__name__)


class MyEmbeddings(BaseRagasEmbeddings):
    """自定义 embedding 类"""

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


def run_rag_eval(
    questions: List[str],
    ground_truths: List[str],
    mode_name: str,
    mode_config: Dict[str, Any],
) -> Dict[str, Any]:
    """运行单个模式的 RAG 评估"""
    logger.info(f"开始评估模式: {mode_name}")

    answers = []
    contexts_list = []

    # 对每个问题运行 RAG pipeline
    for i, question in enumerate(questions):
        logger.info(f"  处理问题 {i + 1}/{len(questions)}: {question[:30]}...")
        answer, contexts = rag_pipeline(
            question,
            return_contexts=True,
            **mode_config,
        )
        answers.append(answer)
        contexts_list.append(contexts)

    # 构建 RAGAS 数据集
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)

    # 初始化 embedding 和 llm
    embeddings = MyEmbeddings()
    llm = llm_factory(
        config.openai.chat,
        client=OpenAI(api_key=config.openai.chat_api, base_url=config.openai.chat_url),
        extra_body={
            "reasoning_split": True,
            "response_format": {"type": "json_object"},
            "raw_json": True,
            "reasoning_effort": None,
            "reasoning": False,
            "max_tokens": 8192,
        },
    )

    # 运行评估
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,  # type: ignore
        embeddings=embeddings,
    )

    logger.info(f"模式 {mode_name} 评估完成")
    return {
        "mode": mode_name,
        "scores": result.scores,  # type: ignore
        "details": result.to_pandas().to_dict("records"),  # type: ignore
    }


# 7种评估模式配置
MODES = {
    "1. base (without web)": {
        "modes": {"base"},
        "use_web": False,
        "use_rerank": "transformers",
        "topk": 3,
        "max_context_length": 1000,
    },
    "2. base (with web)": {
        "modes": {"base"},
        "use_web": True,
        "use_rerank": "transformers",
        "topk": 3,
        "max_context_length": 1000,
    },
    "3. base + mqe": {
        "modes": {"base", "mqe"},
        "use_web": True,
        "mqe_n": 3,
        "topk": 3,
        "max_context_length": 1000,
    },
    "4. base + hyde (完整)": {
        "modes": {"base", "hyde"},
        "use_web": True,
        "hyde_n": 3,
        "topk": 3,
        "max_context_length": 1000,
    },
    "5. base + hyde (仅 hypo_doc)": {
        "modes": {"base", "hyde"},
        "use_web": True,
        "use_hypo_doc": True,
        "use_hyde_rewrite": False,
        "topk": 3,
        "max_context_length": 1000,
    },
    "6. base + hyde (仅 hyde_rewrite)": {
        "modes": {"base", "hyde"},
        "use_web": True,
        "use_hypo_doc": False,
        "use_hyde_rewrite": True,
        "topk": 3,
        "max_context_length": 1000,
    },
    "7. base + mqe + hyde (完整)": {
        "modes": {"base", "mqe", "hyde"},
        "use_web": True,
        "mqe_n": 3,
        "hyde_n": 3,
        "topk": 3,
        "max_context_length": 1000,
    },
}


def main():
    # 加载测试数据
    logger.info("加载测试数据集...")
    test_data = load_test_dataset()
    questions = [item["question"] for item in test_data]
    ground_truths = [item["ground_truth"] for item in test_data]

    logger.info(f"测试数据集包含 {len(questions)} 个问题")

    # 存储所有结果
    all_results = []

    # 运行7种模式的评估
    for mode_name, mode_config in MODES.items():
        try:
            result = run_rag_eval(
                questions=questions,
                ground_truths=ground_truths,
                mode_name=mode_name,
                mode_config=mode_config,
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"模式 {mode_name} 评估失败: {e}")
            all_results.append(
                {
                    "mode": mode_name,
                    "error": str(e),
                }
            )

    # 打印结果汇总
    logger.info("\n" + "=" * 80)
    logger.info("评估结果汇总")
    logger.info("=" * 80)

    for result in all_results:
        if "error" in result:
            logger.info(f"{result['mode']}: ERROR - {result['error']}")
        else:
            scores = result["scores"][0]  # scores 是一个列表，取第一个元素
            logger.info(f"{result['mode']}:")
            logger.info(f"  faithfulness: {scores['faithfulness']:.4f}")
            logger.info(f"  answer_relevancy: {scores['answer_relevancy']:.4f}")
            logger.info(f"  context_precision: {scores['context_precision']:.4f}")
            logger.info(f"  context_recall: {scores['context_recall']:.4f}")

    # 保存详细结果到文件
    output_path = "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"\n详细结果已保存到 {output_path}")


if __name__ == "__main__":
    main()
