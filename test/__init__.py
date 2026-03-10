# -*- coding: utf-8 -*-
"""
测试模块

包含:
- dataset_generator: 数据集生成器
- ragas_evaluator: RAGAS评估器

使用方式:
    from test import generate_test_dataset, load_test_dataset

    # 生成测试集
    samples = generate_test_dataset(n_samples=10)

    # 加载测试集
    samples = load_test_dataset()

    # 评估
    from test import simple_evaluate
    results = simple_evaluate(samples)
"""

from .dataset_generator import generate_test_dataset, load_test_dataset, TestSample

# RAGAS评估器 - 延迟导入以避免不必要的模型加载
def __getattr__(name):
    if name in ["evaluate_with_ragas", "simple_evaluate", "rag_pipeline", "EvaluationResult", "RAGAS_AVAILABLE"]:
        from . import ragas_evaluator
        return getattr(ragas_evaluator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "generate_test_dataset",
    "load_test_dataset",
    "TestSample",
    "evaluate_with_ragas",
    "simple_evaluate",
    "rag_pipeline",
    "EvaluationResult",
    "RAGAS_AVAILABLE",
]
