# -*- coding: utf-8 -*-
"""
评估模块

使用方式:
    from eval import generate_test_dataset, load_test_dataset

    # 生成测试集
    samples = generate_test_dataset(n_samples=10)

    # 加载测试集
    samples = load_test_dataset()

    # 评估 (运行 python -m eval.run_eval)
"""

from .gen_testset import TestSample, load_test_dataset, generate_test_dataset

__all__ = [
    "TestSample",
    "load_test_dataset",
    "generate_test_dataset",
]
