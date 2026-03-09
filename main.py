import logging
from config import config, CHAT_MODEL, EMBED_MODEL
from rerank.llm import llm_cross_encoder_rerank

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # 数据预处理
    # ingest()

    query = "哪个分析师做了顺丰的研报？"
    embedding = EMBED_MODEL(query)
    # docs = hybrid_retriever(query, 10)
    # logging.info(f"检索到 {len(docs)} 条相关文档，准备重排序...")
    # docs = llm_cross_encoder_rerank(query, docs, 5)
    # logging.info(f"重排序完成，最终选取 {len(docs)} 条文档作为回答依据。")

    # context = "\n\n".join(d.page_content for d in docs)
    # prompt = f"""
    # 请根据以下内容回答问题
    # {context}

    # 问题:
    # {query}
    # """
    # ans = CHAT_MODEL.invoke(prompt).content
    # logger.info("\n回答:\n", ans)
