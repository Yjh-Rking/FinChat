import logging
from typing import Set, Optional, Union, Tuple, List
from core.retriever import HybridSearcher
from core.rerank import (
    llm_cross_encoder_rerank,
    transformers_cross_encoder_rerank,
    reciprocal_rank_fusion,
)
from core.api import chat_model

logger = logging.getLogger(__name__)


def rag_pipeline(
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
    return_contexts: bool = False,
    max_context_length: Optional[int] = None,
) -> Union[str, Tuple[str, List[str]]]:
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
        return_contexts: 是否返回检索到的上下文列表
        max_context_length: 上下文最大字符数，None 表示不限制

    Returns:
        如果 return_contexts=False: LLM 生成的答案 (str)
        如果 return_contexts=True: (答案, 上下文列表) (tuple)
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
    logger.debug(f"{len(all_retrieval_results)} group retrieval results for RRF fusion")
    docs = reciprocal_rank_fusion(all_retrieval_results, k=rrf_k)
    logger.debug(f"{len(docs)} docs after RRF fusion")

    # 重排序阶段
    if use_rerank != "none" and docs:
        if use_rerank == "transformers":
            docs = transformers_cross_encoder_rerank(query, docs, topk=topk)
        elif use_rerank == "llm":
            docs = llm_cross_encoder_rerank(query, docs, topk=topk)
        logger.debug(f"{len(docs)} docs after rerank")

    # 获取上下文文本列表
    contexts = [doc.text for doc in docs]

    # 限制上下文长度
    if max_context_length is not None and contexts:
        truncated_contexts = []
        total_length = 0
        for ctx in contexts:
            if total_length + len(ctx) + 2 <= max_context_length:
                truncated_contexts.append(ctx)
                total_length += len(ctx) + 2  # +2 for "\n\n"
            else:
                break
        contexts = truncated_contexts

    # LLM 生成答案
    if contexts:
        context = "\n\n".join(contexts)
        prompt = f"""请根据以下内容回答问题。
            ---
            {context}
            ---
            问题: {query}
            要求：简洁准确地回答，直接给出答案，不需要解释。
        """
        answer = chat_model(
            prompt=prompt, system_message="你是一个专业的金融分析师助手。"
        )
    else:
        answer = "抱歉，没有找到相关信息。"

    if return_contexts:
        return answer, contexts
    return answer
