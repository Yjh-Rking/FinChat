"""
混合检索模块

支持多种检索模式：
1. base - 基础检索 (vector + BM25 + web)
2. mqe - Multi-Query Enhancement
3. hyde - HyDE 查询增强

最后统一 RRF 融合所有检索结果
"""

import logging
from typing import List, Set, Optional
from core.models import RetrievalDoc, SQLiteStore, QdrantStore
from config import config

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    混合检索器

    支持多种检索模式，可组合使用：
    - base: 基础检索 (vector + BM25 + web)
    - mqe: Multi-Query Enhancement
    - hyde: HyDE 查询增强

    所有检索结果统一 RRF 融合
    """

    def __init__(self):
        """初始化检索器"""
        self.sqlite_store = SQLiteStore(config.sqlite.path)
        self.qdrant_store = QdrantStore(
            url=config.qdrant.url, collection=config.qdrant.collection
        )
        self.all_chunks = self.sqlite_store.get_all_chunks()

        # 延迟导入避免循环依赖
        from core.retriever import VectorRetriever, BM25Retriever, TavilyWebRetriever
        from core.query import generate_hypo_docs, hyde_rewrite, multiquery_enhance

        self.vector_retriever = VectorRetriever(self.sqlite_store)
        self.bm25_retriever = BM25Retriever(self.all_chunks, language="zh")
        self.web_retriever = TavilyWebRetriever(api_key=config.tavily.token)

        # 查询增强函数
        self.generate_hypo_docs = generate_hypo_docs
        self.hyde_rewrite = hyde_rewrite
        self.multiquery_enhance = multiquery_enhance

        # RRF 函数
        from core.rerank import reciprocal_rank_fusion

        self.rrf = reciprocal_rank_fusion

    # ==================== 基础检索方法 ====================

    def vector_retrieve(self, query: str, topk: int = 10) -> List[RetrievalDoc]:
        """向量检索"""
        return self.vector_retriever.retrieve(
            query, topk=topk, qdrant=self.qdrant_store
        )

    def bm25_retrieve(self, query: str, topk: int = 10) -> List[RetrievalDoc]:
        """BM25 检索"""
        return self.bm25_retriever.retrieve(query, topk=topk)

    def web_retrieve(self, query: str, topk: int = 10) -> List[RetrievalDoc]:
        """Web 检索"""
        return self.web_retriever.retrieve(query, topk=topk)

    # ==================== 检索流程 ====================

    def base_flow(
        self,
        query: str,
        use_web: bool = True,
        topk: int = 10,
    ) -> List[List[RetrievalDoc]]:
        """
        基础流程: vector + BM25 + (optional) web

        Returns:
            检索结果列表 (每组是一个检索器的结果，不进行 RRF)
        """
        all_docs = []

        # 向量检索
        vector_docs = self.vector_retrieve(query, topk)
        logger.debug(f"Base - Vector: {len(vector_docs)} docs")
        all_docs.append(vector_docs)

        # BM25 检索
        bm25_docs = self.bm25_retrieve(query, topk)
        logger.debug(f"Base - BM25: {len(bm25_docs)} docs")
        all_docs.append(bm25_docs)

        # Web 检索 (可选)
        if use_web:
            web_docs = self.web_retrieve(query, topk)
            logger.debug(f"Base - Web: {len(web_docs)} docs")
            all_docs.append(web_docs)

        return all_docs

    def mqe_flow(
        self,
        query: str,
        n: int = 5,
        topk: int = 10,
    ) -> List[List[RetrievalDoc]]:
        """
        MQE 流程: multiquery -> vector + BM25

        Returns:
            检索结果列表 (不进行 RRF)
        """
        # 生成多个查询
        queries = self.multiquery_enhance(query, n=n, original_query=True)
        logger.debug(f"MQE retriever: {queries}")

        # 每个查询分别进行 vector + BM25 检索
        all_docs = []
        for q in queries:
            vector_docs = self.vector_retrieve(q, topk)
            bm25_docs = self.bm25_retrieve(q, topk)
            all_docs.append(vector_docs)
            all_docs.append(bm25_docs)

        return all_docs

    def hyde_flow(
        self,
        query: str,
        use_hypo_doc: bool = True,
        use_hyde_rewrite: bool = True,
        hyde_n: int = 5,
        topk: int = 10,
    ) -> List[List[RetrievalDoc]]:
        """
        HyDE 流程:
        - use_hypo_doc: query -> hyde_doc -> embedding -> vector retriever
        - use_hyde_rewrite: query -> hyde_rewrite -> vector + BM25 retriever

        Returns:
            检索结果列表 (不进行 RRF)
        """
        all_docs = []

        # 1. 假设文档检索 (可选)
        if use_hypo_doc:
            hypo_doc = self.generate_hypo_docs(query)
            logger.debug(f"HyDE docs: {hypo_doc[:50]}...")
            hypo_vector_docs = self.vector_retrieve(hypo_doc, topk)
            logger.debug(f"HyDE vector retriever: {len(hypo_vector_docs)} docs")
            all_docs.append(hypo_vector_docs)

        # 2. HyDE rewrite 检索 (可选)
        if use_hyde_rewrite:
            rewrite_queries = self.hyde_rewrite(query, n=hyde_n, original_query=True)
            logger.debug(f"HyDE rewrite retriever: {rewrite_queries}")

            for q in rewrite_queries:
                vector_docs = self.vector_retrieve(q, topk)
                bm25_docs = self.bm25_retrieve(q, topk)
                all_docs.append(vector_docs)
                all_docs.append(bm25_docs)

        return all_docs

    # ==================== 主检索方法 ====================

    def retrieve(
        self,
        query: str,
        modes: Optional[Set[str]] = None,
        use_web: bool = True,
        topk: int = 10,
        mqe_n: int = 5,
        hyde_n: int = 5,
        use_hypo_doc: bool = True,
        use_hyde_rewrite: bool = True,
    ) -> List[List[RetrievalDoc]]:
        """
        混合检索方法 - 返回所有检索结果（不进行 RRF）

        Args:
            query: 用户查询
            modes: 查询模式集合 - {"base", "mqe", "hyde"}，默认 {"base"}
            use_web: 是否使用 web 检索 (仅 base 模式有效)
            topk: 检索返回数量
            mqe_n: MQE 模式生成的查询数量
            hyde_n: HyDE 模式 rewrite 生成的查询数量
            use_hypo_doc: 是否使用假设文档检索 (仅 hyde 模式有效)
            use_hyde_rewrite: 是否使用 hyde rewrite 检索 (仅 hyde 模式有效)

        Returns:
            检索结果列表 (List[List[RetrievalDoc]])，每组是一个检索器的结果
        """
        if modes is None:
            modes = {"base"}

        logger.debug(f"retrieval modes: {modes}")
        all_retrieval_results = []

        # 收集各模式检索结果
        if "base" in modes:
            results = self.base_flow(query, use_web=use_web, topk=topk)
            all_retrieval_results.extend(results)
            logger.debug(f"Base process: {len(results)} groups of retrieval results")

        if "mqe" in modes:
            results = self.mqe_flow(query, n=mqe_n, topk=topk)
            all_retrieval_results.extend(results)
            logger.debug(
                f"MQE process completed, collected {len(results)} groups of retrieval results"
            )

        if "hyde" in modes:
            results = self.hyde_flow(
                query,
                use_hypo_doc=use_hypo_doc,
                use_hyde_rewrite=use_hyde_rewrite,
                hyde_n=hyde_n,
                topk=topk,
            )
            all_retrieval_results.extend(results)
            logger.debug(
                f"HyDE process completed, collected {len(results)} groups of retrieval results"
            )

        return all_retrieval_results
