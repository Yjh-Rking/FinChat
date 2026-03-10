"""
RAG 查询主流程

使用 HybridSearcher 进行混合检索，在 main 中进行 RRF 融合，最后重排序并生成答案
"""

import logging
from core.rag import process_query

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    query = "哪个券商哪个研究员写了顺丰的研报？研报的主要观点是什么？"
    topk = 5
    # 1. 仅 base (with web)
    print("=" * 50)
    print("模式: base (with web)")
    answer = process_query(
        query, modes={"base"}, use_web=True, use_rerank="transformers", topk=topk
    )
    print(f"问题: {query}")
    print(f"答案: {answer}")

    # 2. 仅 base (without web)
    print("=" * 50)
    print("模式: base (without web)")
    answer = process_query(
        query, modes={"base"}, use_web=False, use_rerank="transformers", topk=topk
    )
    print(f"问题: {query}")
    print(f"答案: {answer}")

    # 3. base + mqe
    print("=" * 50)
    print("模式: base + mqe")
    answer = process_query(
        query, modes={"base", "mqe"}, use_web=True, mqe_n=3, topk=topk
    )
    print(f"问题: {query}")
    print(f"答案: {answer}")

    # 4. base + hyde (完整)
    print("=" * 50)
    print("模式: base + hyde (完整)")
    answer = process_query(
        query, modes={"base", "hyde"}, use_web=True, hyde_n=3, topk=topk
    )
    print(f"问题: {query}")
    print(f"答案: {answer}")

    # 5. base + hyde (仅 hyde_doc)
    print("=" * 50)
    print("模式: base + hyde (仅 hypo_doc)")
    answer = process_query(
        query,
        modes={"base", "hyde"},
        use_web=True,
        use_hypo_doc=True,
        use_hyde_rewrite=False,
        topk=topk,
    )
    print(f"问题: {query}")
    print(f"答案: {answer}")

    # 6. base + hyde (仅 hyde_rewrite)
    print("=" * 50)
    print("模式: base + hyde (仅 hyde_rewrite)")
    answer = process_query(
        query,
        modes={"base", "hyde"},
        use_web=True,
        use_hypo_doc=False,
        use_hyde_rewrite=True,
        topk=topk,
    )
    print(f"问题: {query}")
    print(f"答案: {answer}")

    # 7. base + mqe + hyde (完整)
    print("=" * 50)
    print("模式: base + mqe + hyde (完整)")
    answer = process_query(
        query, modes={"base", "mqe", "hyde"}, use_web=True, mqe_n=3, hyde_n=3, topk=topk
    )
    print(f"问题: {query}")
    print(f"答案: {answer}")
