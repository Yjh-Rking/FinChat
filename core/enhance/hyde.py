from typing import List, Any, Dict, Tuple
from core.config import CHAT_MODEL

SYSTEM_PROMPT = (
    "你是中文知识库助手。"
    "请根据用户问题生成一段可能出现在知识库中的中文说明文，用于检索，不要编造具体数字。"
)
USER_PROMPT = """请围绕这个问题写一段"可能的参考答案文本"（用于检索召回）：
    问题：{question}

    要求：
    1) 简体中文
    2) 120~300字
    3) 包含关键术语、同义表达、实现步骤词（如"配置/示例/参数/兼容/版本差异"）
    4) 不要输出思考过程或标签
    5) 只输出正文"""


def hyde_generate_hypo_doc(question: str, max_chars: int = 800) -> str:
    """生成 HyDE 的 hypothetical document（中文假设答案/说明文）。"""
    text = CHAT_MODEL(
        prompt=USER_PROMPT.format(question=question),
        system_message=SYSTEM_PROMPT,
    )
    return text[:max_chars].strip()


def _doc_key(doc: Any) -> str:
    meta = getattr(doc, "metadata", {}) or {}
    if "id" in meta:
        return f"id:{meta['id']}"
    src = meta.get("source", "")
    content = getattr(doc, "page_content", "") or ""
    return f"{src}::{hash(content)}"


def reciprocal_rank_fusion(ranked_lists: List[List[Any]], k: int = 60) -> List[Any]:
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Any] = {}

    for docs in ranked_lists:
        for rank, doc in enumerate(docs, start=1):
            key = _doc_key(doc)
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    fused: List[Tuple[str, float]] = sorted(
        scores.items(), key=lambda x: x[1], reverse=True
    )
    return [doc_map[key] for key, _ in fused]


def retrieve(retriever, query: str):
    """兼容 invoke / get_relevant_documents。"""
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    return retriever.get_relevant_documents(query)


def hyde_retrieve(
    retriever,
    question: str,
    final_top_k: int = 10,
    fuse_with_original: bool = True,
):
    """
    HyDE 检索：
    - 用"假设文档"检索
    - 可选融合"原问题"检索结果
    """
    hypo_doc = hyde_generate_hypo_doc(question)
    docs_hyde = retrieve(retriever, hypo_doc)

    if fuse_with_original:
        docs_q = retrieve(retriever, question)
        fused = reciprocal_rank_fusion([docs_q, docs_hyde], k=60)
        return {
            "hypo_doc": hypo_doc,
            "docs": fused[:final_top_k],
            "debug": {
                "orig_count": len(docs_q or []),
                "hyde_count": len(docs_hyde or []),
            },
        }

    return {
        "hypo_doc": hypo_doc,
        "docs": (docs_hyde or [])[:final_top_k],
        "debug": {"hyde_count": len(docs_hyde or [])},
    }


# if __name__ == "__main__":
#     retriever = get_vector_retriever(10)
#     q = "langchain v1.2.10 里怎么实现多重查询检索？"
#     result = hyde_retrieve(
#         retriever=retriever,
#         question=q,
#         final_top_k=10,
#         fuse_with_original=True,
#     )

#     print("HyDE 假设文档：\n", result["hypo_doc"])
#     print("Debug:", result["debug"])
#     print("Top docs:", len(result["docs"]))
#     for i, d in enumerate(result["docs"][:10], 1):
#         print(f"[{i}] {d.page_content[:120]}...")
