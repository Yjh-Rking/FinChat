# -*- coding: utf-8 -*-
"""
RAG评估系统运行脚本

用法:
    python test/run_evaluation.py --generate          # 生成测试数据集
    python test/run_evaluation.py --evaluate         # 执行评估
    python test/run_evaluation.py --all              # 生成数据集并执行评估
"""

import argparse
import sys

# 添加项目根目录到路径
sys.path.insert(0, "/home/rking/FinChat")

from test import (
    generate_test_dataset,
    load_test_dataset,
    evaluate_with_ragas,
    simple_evaluate,
    RAGAS_AVAILABLE,
)


def main():
    parser = argparse.ArgumentParser(description="RAG评估系统")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="生成测试数据集",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="执行评估",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="生成数据集并执行评估",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="生成的测试样本数量",
    )
    parser.add_argument(
        "--use-ragas",
        action="store_true",
        help="使用RAGAS进行评估（需要安装ragas）",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/test_dataset.json",
        help="测试数据集路径",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/evaluation_results.json",
        help="评估结果输出路径",
    )

    args = parser.parse_args()

    # 默认执行所有操作
    if not (args.generate or args.evaluate or args.all):
        args.all = True

    if args.all:
        args.generate = True
        args.evaluate = True

    # 1. 生成测试数据集
    if args.generate:
        print("=" * 50)
        print("生成测试数据集")
        print("=" * 50)
        samples = generate_test_dataset(
            n_samples=args.n_samples,
            output_path=args.dataset_path,
            use_llm=True,
        )
        print(f"生成了 {len(samples)} 条测试样本\n")

    # 2. 执行评估
    if args.evaluate:
        print("=" * 50)
        print("执行评估")
        print("=" * 50)

        # 检查RAGAS是否可用
        if args.use_ragas and not RAGAS_AVAILABLE:
            print("Warning: RAGAS not available, falling back to simple evaluation")
            print("Install RAGAS with: pip install ragas")
            args.use_ragas = False

        # 加载测试数据集
        samples = load_test_dataset(args.dataset_path)
        print(f"加载了 {len(samples)} 条测试样本\n")

        if args.use_ragas:
            # 使用RAGAS评估
            results = evaluate_with_ragas(
                samples,
                output_path=args.output_path,
            )
        else:
            # 简单评估
            results = simple_evaluate(
                samples,
                output_path=args.output_path,
            )

        print(f"\n评估完成，共评估 {len(results)} 个样本")


if __name__ == "__main__":
    main()
