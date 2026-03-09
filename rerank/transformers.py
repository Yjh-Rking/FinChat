from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")


def transformers_rerank(query, docs, topk=5):
    reranker = CrossEncoder("BAAI/bge-reranker-base")
    pairs = [(query, d.text) for d in docs]
    scores = reranker.predict(pairs)
    for doc, score in zip(docs, scores):
        doc.score = float(score)
    docs = sorted(docs, key=lambda x: x.score, reverse=True)
    return docs[:topk]
