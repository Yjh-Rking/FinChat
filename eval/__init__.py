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

from .gen_testset import TestSample, load_test_dataset, generate_test_dataset

__all__ = [
    "TestSample",
    "load_test_dataset",
    "generate_test_dataset",
]
