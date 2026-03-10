# -*- coding: utf-8 -*-
"""
RAGAS评估器

使用RAGAS框架对RAG系统进行评估。
支持多种评估指标：Faithfulness, Answer Relevance, Context Precision, Context Recall
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

# RAGAS依赖
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("Warning: ragas not installed. Install with: pip install ragas")

from core.config import config


@dataclass
class EvaluationResult:
    """评估结果"""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    answer_similarity: Optional[float] = None


def rag_pipeline(
    query: str, topk: int = 5, use_rerank: bool = False
) -> tuple[str, list[str]]:
    """
    执行RAG pipeline，返回答案和检索到的上下文

    Args:
        query: 用户查询
        topk: 检索返回数量
        use_rerank: 是否使用重排序（需要下载模型）

    Returns:
        (答案, 上下文列表)
    """
    from core.retriever import HybridSearcher
    from core.rerank import reciprocal_rank_fusion
    from core.config import CHAT_MODEL

    # 初始化混合检索器
    searcher = HybridSearcher()
    all_retrieval_results = searcher.retrieve(
        query=query,
        modes={"base"},
        use_web=False,
        topk=topk,
    )

    # RRF融合
    docs = reciprocal_rank_fusion(all_retrieval_results, k=60)
    docs = docs[: topk * 2]

    # 重排序（可选）
    if use_rerank:
        from core.rerank import transformers_cross_encoder_rerank

        docs = transformers_cross_encoder_rerank(query, docs, topk=topk)

    # 生成答案
    if docs:
        context = "\n\n".join(doc.text for doc in docs)
        contexts = [doc.text for doc in docs]

        prompt = f"""请根据以下内容回答问题。

---
{context}
---

问题: {query}

要求：简洁准确地回答，直接给出答案，不需要解释。"""

        answer = CHAT_MODEL(
            prompt=prompt,
            system_message="你是一个专业的金融分析师助手。",
        )
    else:
        answer = "抱歉，没有找到相关信息。"
        contexts = []

    return answer, contexts


def evaluate_with_ragas(
    samples: list,
    output_path: str = "data/evaluation_results.json",
    metrics: Optional[list] = None,
) -> list[EvaluationResult]:
    """
    使用RAGAS进行评估

    Args:
        samples: 测试样本列表 (TestSample对象)
        output_path: 输出结果路径
        metrics: 评估指标列表，默认使用faithfulness, answer_relevancy, context_precision

    Returns:
        评估结果列表
    """
    if not RAGAS_AVAILABLE:
        print("RAGAS not available, please install it first")
        return []

    if metrics is None:
        # 默认使用所有5个指标
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_similarity,
        ]

    # 准备评估数据
    eval_data = []
    for sample in samples:
        # 执行RAG pipeline
        answer, contexts = rag_pipeline(sample.question)

        eval_data.append(
            {
                "question": sample.question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": sample.ground_truth,
            }
        )

    # 初始化LLM和Embedding
    llm = ChatOpenAI(
        model=config.chat.model,
        api_key=config.chat.token,  # type: ignore
        base_url=config.chat.url,
    )
    embedding = OpenAIEmbeddings(
        model=config.embed.model,
        api_key=config.embed.token,  # type: ignore
        base_url="https://api.siliconflow.cn/v1/",
    )

    # 包装为RAGAS格式
    llm_wrapper = LangchainLLMWrapper(llm)
    embed_wrapper = LangchainEmbeddingsWrapper(embedding)

    # 创建评估数据集
    from datasets import Dataset

    dataset = Dataset.from_list(eval_data)

    # 执行评估
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm_wrapper,
        embeddings=embed_wrapper,
    )

    # 整理结果
    eval_results = []
    for i, row in enumerate(results):  # type: ignore
        result = EvaluationResult(
            question=eval_data[i]["question"],
            answer=eval_data[i]["answer"],
            contexts=eval_data[i]["contexts"],
            ground_truth=eval_data[i]["ground_truth"],
            faithfulness=row.get("faithfulness"),
            answer_relevancy=row.get("answer_relevancy"),
            context_precision=row.get("context_precision"),
            context_recall=row.get("context_recall"),
            answer_similarity=row.get("answer_similarity"),
        )
        eval_results.append(result)

    # 保存结果
    output_data = [asdict(r) for r in eval_results]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # 打印汇总
    print("\n" + "=" * 50)
    print("评估结果汇总")
    print("=" * 50)

    metric_scores = {}
    for metric in [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_similarity",
    ]:
        scores = [
            getattr(r, metric) for r in eval_results if getattr(r, metric) is not None
        ]
        if scores:
            avg = sum(scores) / len(scores)
            metric_scores[metric] = avg
            print(f"{metric}: {avg:.4f}")

    print("=" * 50)
    print(f"详细结果已保存到: {output_path}")

    return eval_results


def simple_evaluate(
    samples: list,
    output_path: str = "data/simple_evaluation_results.json",
    use_rerank: bool = False,
) -> list[EvaluationResult]:
    """
    简单的评估方式（不依赖RAGAS），仅执行RAG pipeline并返回结果

    适用于无法安装RAGAS的场景。
    """
    eval_results = []

    for i, sample in enumerate(samples):
        print(f"处理样本 {i + 1}/{len(samples)}: {sample.question[:50]}...")

        # 执行RAG pipeline
        answer, contexts = rag_pipeline(sample.question, use_rerank=use_rerank)

        result = EvaluationResult(
            question=sample.question,
            answer=answer,
            contexts=contexts,
            ground_truth=sample.ground_truth,
        )
        eval_results.append(result)

    # 保存结果
    output_data = [asdict(r) for r in eval_results]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"评估完成，结果保存到: {output_path}")
    return eval_results


if __name__ == "__main__":
    # 测试
    from test.dataset_generator import load_test_dataset

    # 加载测试数据集
    samples = load_test_dataset("data/test_dataset.json")

    # 使用简单评估（不需要RAGAS依赖）
    results = simple_evaluate(samples[:3])
    print(f"评估了 {len(results)} 个样本")
