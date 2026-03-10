"""
RAG 查询主流程

使用 HybridSearcher 进行混合检索，在 main 中进行 RRF 融合，最后重排序并生成答案
"""

import logging
from typing import Set, Optional
from core.config import CHAT_MODEL
from core.retriever import HybridSearcher
from core.rerank import (
    llm_cross_encoder_rerank,
    transformers_cross_encoder_rerank,
    reciprocal_rank_fusion,
)

logger = logging.getLogger(__name__)


def process_query(
    query: str,
    modes: Optional[Set[str]] = None,
    use_web: bool = True,
    use_rerank: str = "transformers",
    topk: int = 10,
    mqe_n: int = 5,
    hyde_n: int = 5,
    use_hypo_doc: bool = True,
    use_hyde_rewrite: bool = True,
    rrf_k: int = 60,
) -> str:
    """
    处理查询的统一入口

    Args:
        query: 用户查询
        modes: 查询模式集合 - {"base", "mqe", "hyde"}，默认 {"base"}
        use_web: 是否使用 web 检索 (仅 base 模式有效)
        use_rerank: 重排序方法 - "transformers", "llm", "none"
        topk: 检索返回数量
        mqe_n: MQE 模式生成的查询数量
        hyde_n: HyDE 模式 rewrite 生成的查询数量
        use_hypo_doc: 是否使用假设文档检索 (仅 hyde 模式有效)
        use_hyde_rewrite: 是否使用 hyde rewrite 检索 (仅 hyde 模式有效)
        rrf_k: RRF 融合参数

    Returns:
        LLM 生成的答案
    """
    if modes is None:
        modes = {"base"}

    # 初始化混合检索器并执行检索
    searcher = HybridSearcher()
    all_retrieval_results = searcher.retrieve(
        query=query,
        modes=modes,
        use_web=use_web,
        topk=topk,
        mqe_n=mqe_n,
        hyde_n=hyde_n,
        use_hypo_doc=use_hypo_doc,
        use_hyde_rewrite=use_hyde_rewrite,
    )

    # RRF 融合
    logger.info(f"总共 {len(all_retrieval_results)} 组检索结果进行 RRF 融合")
    docs = reciprocal_rank_fusion(all_retrieval_results, k=rrf_k)
    logger.info(f"RRF 融合后: {len(docs)} 文档")

    # 重排序阶段
    if use_rerank != "none" and docs:
        if use_rerank == "transformers":
            docs = transformers_cross_encoder_rerank(query, docs, topk=topk)
        elif use_rerank == "llm":
            docs = llm_cross_encoder_rerank(query, docs, topk=topk)
        logger.info(f"重排序完成，剩余 {len(docs)} 文档")

    # LLM 生成答案
    if docs:
        context = "\n\n".join(doc.text for doc in docs)
        prompt = f"""请根据以下内容回答问题。

---
{context}
---

问题: {query}

要求：简洁准确地回答，直接给出答案，不需要解释。"""
        answer = CHAT_MODEL(
            prompt=prompt, system_message="你是一个专业的金融分析师助手。"
        )
    else:
        answer = "抱歉，没有找到相关信息。"

    return answer


if __name__ == "__main__":
    query = "哪个券商哪个研究员写了顺丰的研报？研报的主要观点是什么？"

    # 1. 仅 base (with web)
    print("=" * 50)
    print("模式: base (with web)")
    answer = process_query(
        query, modes={"base"}, use_web=True, use_rerank="transformers"
    )
    print(f"问题: {query}")
    print(f"答案: {answer}")

    # 2. 仅 base (without web)
    print("=" * 50)
    print("模式: base (without web)")
    answer = process_query(
        query, modes={"base"}, use_web=False, use_rerank="transformers"
    )
    print(f"问题: {query}")
    print(f"答案: {answer}")

    # 3. base + mqe
    print("=" * 50)
    print("模式: base + mqe")
    answer = process_query(query, modes={"base", "mqe"}, use_web=True, mqe_n=3)
    print(f"问题: {query}")
    print(f"答案: {answer}")

    # 4. base + hyde (完整)
    print("=" * 50)
    print("模式: base + hyde (完整)")
    answer = process_query(query, modes={"base", "hyde"}, use_web=True, hyde_n=3)
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
    )
    print(f"问题: {query}")
    print(f"答案: {answer}")

    # 7. base + mqe + hyde (完整)
    print("=" * 50)
    print("模式: base + mqe + hyde (完整)")
    answer = process_query(
        query, modes={"base", "mqe", "hyde"}, use_web=True, mqe_n=3, hyde_n=3
    )
    print(f"问题: {query}")
    print(f"答案: {answer}")
