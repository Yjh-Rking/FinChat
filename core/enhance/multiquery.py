import re
from core.config import CHAT_MODEL

SYSTEM_PROMPT = "你是中文检索查询改写助手。请严格输出多条用于向量检索的查询语句。"
USER_PROMPT = (
    "请把下面问题改写为 {n} 条不同角度的中文检索查询。"
    "要求：\n"
    "1) 全部简体中文\n"
    "2) 每行一条，不要编号\n"
    "3) 不要解释，不要多余文本\n"
    "4) 保留原问题中的实体、时间、约束\n\n"
    "原问题：{question}"
)


def generate_multi_queries(
    question,
    n,
):
    text = CHAT_MODEL(
        prompt=USER_PROMPT.format(n=n, question=question),
        system_message=SYSTEM_PROMPT,
    )

    lines = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^[\-\d\.\)\s]+", "", line)  # 去掉可能的编号
        if line:
            lines.append(line)

    # 兜底：把原问题放进去，避免改写跑偏
    if question not in lines:
        lines = [question] + lines

    # 去重并截断
    dedup = []
    seen = set()
    for q in lines:
        if q not in seen:
            seen.add(q)
            dedup.append(q)

    return dedup[: max(n, 1)]


def _doc_key(doc) -> str:
    """尽量稳定地给文档做去重 key。"""
    meta = getattr(doc, "metadata", {}) or {}
    if "id" in meta:
        return f"id:{meta['id']}"
    src = meta.get("source", "")
    content = getattr(doc, "page_content", "") or ""
    return f"{src}::{hash(content)}"


def reciprocal_rank_fusion(ranked_lists, k):
    """RRF 融合多个检索结果列表。"""
    scores = {}
    doc_map = {}

    for docs in ranked_lists:
        for rank, doc in enumerate(docs, start=1):
            key = _doc_key(doc)
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in fused]


def multi_query_retrieve(
    retriever,
    question,
    num_queries: int = 5,
    final_top_k: int = 12,
):
    # 1) 生成多查询
    queries = generate_multi_queries(question, n=num_queries)

    # 2) 每个 query 分别检索
    all_ranked_docs = []
    for q in queries:
        # 兼容不同 retriever 接口
        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(q)
        else:
            docs = retriever.get_relevant_documents(q)
        all_ranked_docs.append(docs or [])

    # 3) RRF 融合
    fused_docs = reciprocal_rank_fusion(all_ranked_docs, k=60)

    # 4) 截断
    return queries, fused_docs[:final_top_k]


# if __name__ == "__main__":
#     retriever = get_vector_retriever(5)
#     question = "langchain v1.2.10 里怎么实现多重查询检索？"
#     queries, docs = multi_query_retrieve(
#         retriever=retriever,
#         question=question,
#         num_queries=5,
#         final_top_k=10,
#     )

#     print("生成的子查询：")
#     for i, q in enumerate(queries, 1):
#         print(f"{i}. {q}")

#     print(f"\n融合后文档数: {len(docs)}")
#     for i, d in enumerate(docs[:5], 1):
#         print(f"[{i}] {d.page_content[:120]}...")
