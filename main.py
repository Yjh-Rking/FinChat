"""
RAG 查询主流程

使用 HybridSearcher 进行混合检索，在 main 中进行 RRF 融合，最后重排序并生成答案
"""

import logging
from core.rag import rag_pipeline

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    query = "哪个券商哪个研究员写了顺丰的研报？研报的主要观点是什么？"
    topk = 5

    # 1. 仅 base (without web)
    logger.info("=" * 30 + " mode: base (without web) " + "=" * 30)
    answer = rag_pipeline(
        query, modes={"base"}, use_web=False, use_rerank="transformers", topk=topk
    )
    logger.info(f"\n问题: {query}\n答案: {answer}")

    # 2. 仅 base (with web)
    logger.info("=" * 30 + " mode: base (with web) " + "=" * 30)
    answer = rag_pipeline(
        query, modes={"base"}, use_web=True, use_rerank="transformers", topk=topk
    )
    logger.info(f"\n问题: {query}\n答案: {answer}")

    # 3. base + mqe
    logger.info("=" * 30 + " mode: base + mqe " + "=" * 30)
    answer = rag_pipeline(
        query, modes={"base", "mqe"}, use_web=True, mqe_n=3, topk=topk
    )
    logger.info(f"\n问题: {query}\n答案: {answer}")

    # 4. base + hyde (完整)
    logger.info("=" * 30 + " mode: base + hyde (完整) " + "=" * 30)
    answer = rag_pipeline(
        query, modes={"base", "hyde"}, use_web=True, hyde_n=3, topk=topk
    )
    logger.info(f"\n问题: {query}\n答案: {answer}")

    # 5. base + hyde (仅 hyde_doc)
    logger.info("=" * 30 + " mode: base + hyde (仅 hypo_doc) " + "=" * 30)
    answer = rag_pipeline(
        query,
        modes={"base", "hyde"},
        use_web=True,
        use_hypo_doc=True,
        use_hyde_rewrite=False,
        topk=topk,
    )
    logger.info(f"\n问题: {query}\n答案: {answer}")

    # 6. base + hyde (仅 hyde_rewrite)
    logger.info("=" * 30 + " mode: base + hyde (仅 hyde_rewrite) " + "=" * 30)
    answer = rag_pipeline(
        query,
        modes={"base", "hyde"},
        use_web=True,
        use_hypo_doc=False,
        use_hyde_rewrite=True,
        topk=topk,
    )
    logger.info(f"\n问题: {query}\n答案: {answer}")

    # 7. base + mqe + hyde (完整)
    logger.info("=" * 30 + " mode: base + mqe + hyde (完整) " + "=" * 30)
    answer = rag_pipeline(
        query, modes={"base", "mqe", "hyde"}, use_web=True, mqe_n=3, hyde_n=3, topk=topk
    )
    logger.info(f"\n问题: {query}\n答案: {answer}")
