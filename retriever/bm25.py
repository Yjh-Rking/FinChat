import re
import jieba
from rank_bm25 import BM25Okapi
from typing import List
from models import Chunk, RetrievalDoc


EN_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
}
ZH_STOPWORDS = {
    "的",
    "了",
    "和",
    "是",
    "就",
    "都",
    "而",
    "及",
    "与",
    "或",
    "等",
}


def tokenize(text: str, language: str = "zh") -> List[str]:
    if language == "zh":
        # 中文：使用 jieba 分词
        words = jieba.lcut(text)
        stopwords = ZH_STOPWORDS
        # 过滤：非空、长度>1、非停用词
        tokens = [
            w.strip()
            for w in words
            if w.strip() and len(w.strip()) > 1 and w not in stopwords
        ]
    elif language == "en":
        # 英文：按空格/标点分词 + 小写化
        words = re.findall(r"\b\w+\b", text.lower())
        stopwords = EN_STOPWORDS
        tokens = [w for w in words if w not in stopwords]
    else:
        raise ValueError("Unsupported language. Use 'zh' or 'en'.")

    return tokens


class BM25Retriever:
    def __init__(self, chunks: List[Chunk], language: str = "zh"):
        self.chunks = chunks
        self.language = language

        # 对每个 chunk 的文本进行分词
        tokenized_corpus = [tokenize(c.text, language=language) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, topk: int = 10) -> List[RetrievalDoc]:
        tokenized_query = tokenize(query, language=self.language)
        scores = self.bm25.get_scores(tokenized_query)

        # 获取 top-k 索引
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :topk
        ]

        results = []
        for idx in top_indices:
            c = self.chunks[idx]
            results.append(
                RetrievalDoc(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    text=c.text,
                    score=float(scores[idx]),
                    source="bm25",
                    metadata=c.metadata,
                )
            )
        return results
