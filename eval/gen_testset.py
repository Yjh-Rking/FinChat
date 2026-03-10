# -*- coding: utf-8 -*-
"""
测试数据集生成器

从SQLite中读取随机采样chunks，使用LLM生成问答对
作为RAG评估所需的测试数据集
"""

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

from config import config
from core.models.sqlite import SQLiteStore
from core.api import chat_model


@dataclass
class TestSample:
    question: str
    ground_truth: str
    context: str
    chunk_ids: list[str]


def generate_test_dataset(
    n_samples: int = 20,
    output_path: str = "data/test_dataset.json",
    use_llm: bool = True,
) -> list[TestSample]:
    """
    生成测试数据集

    Args:
        n_samples: 生成的测试样本数量
        output_path: 输出文件路径
        use_llm: 是否使用LLM生成问答对（False则只返回chunk文本）

    Returns:
        测试样本列表
    """
    # 初始化存储
    sqlite_store = SQLiteStore(db_path=config.sqlite.path)

    # 随机采样chunks
    chunks = sqlite_store.get_random_chunks(k=n_samples * 3)  # 多采样一些以便过滤

    # 按doc_id分组，每组只取一个chunk，确保多样性
    grouped_by_doc: dict[str, list] = {}
    for chunk in chunks:
        if chunk.doc_id not in grouped_by_doc:
            grouped_by_doc[chunk.doc_id] = []
        grouped_by_doc[chunk.doc_id].append(chunk)

    # 从每个doc中随机选择一个chunk
    selected_chunks = []
    for doc_chunks in grouped_by_doc.values():
        selected_chunks.append(random.choice(doc_chunks))

    # 截取所需数量
    samples = []
    for chunk in selected_chunks[:n_samples]:
        context = chunk.text

        if use_llm:
            prompt = f"""。
                要求：
                1. 问题必须是关于这份研报内容的具体问题
                2. 答案应该直接来自原文内容
                3. 问题要简洁明了
                4. **请直接返回纯 JSON，不要使用任何代码块或多余文本**
                
                ---
                研报内容：
                {context[:1000]}  # 截取前1000字避免过长
                ---

                请按以下JSON格式输出，不要有其他内容：
                {{
                    "question": "生成的问题",
                    "ground_truth": "标准答案"
                }}
            """

            try:
                response = chat_model(
                    prompt=prompt,
                    system_message="你是一个金融研报分析助手。请根据提供的研报内容生成一个相关的问题和标准答案。",
                    extra_body={
                        "reasoning_split": True,
                        "response_format": {"type": "json_object"},
                        "raw_json": True,  # 关键，告诉模型返回纯 JSON
                    },
                )
                # 尝试解析JSON
                result = json.loads(response)
                question = result.get("question", "")
                ground_truth = result.get("ground_truth", "")
            except Exception as e:
                print(f"LLM生成失败，使用chunk前100字作为ground_truth: {e}")
                question = f"关于{chunk.doc_id}的内容，请总结主要观点"
                ground_truth = context[:200]
        else:
            # 不使用LLM，只返回chunk内容
            question = f"关于{chunk.doc_id}的内容是什么？"
            ground_truth = context[:200]

        sample = TestSample(
            question=question,
            ground_truth=ground_truth,
            context=context,
            chunk_ids=[chunk.chunk_id],
        )
        samples.append(sample)

    # 保存到文件
    output_data = [asdict(s) for s in samples]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"生成了 {len(samples)} 条测试样本，保存到 {output_path}")
    return samples


def load_test_dataset(path: str = "data/test_dataset.json") -> list[TestSample]:
    """加载测试数据集"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [TestSample(**item) for item in data]


if __name__ == "__main__":
    # 生成测试数据集
    samples = generate_test_dataset(n_samples=10, output_path="data/test_dataset.json")
    print(f"生成了 {len(samples)} 条测试样本")
