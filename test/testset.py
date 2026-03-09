import sqlite3
from ragas.testset import TestsetGenerator
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from utils import config, EMBED_MODEL


def load_docs_from_sqlite(sqlite_path, limit=None, doc_id=None):
    docs = []
    with sqlite3.connect(sqlite_path) as conn:
        conn.row_factory = sqlite3.Row  # 允许按列名访问
        sql = "SELECT chunk_id, doc_id, text FROM chunks"
        params = []
        conditions = []

        if doc_id:
            conditions.append("doc_id = ?")
            params.append(doc_id)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        sql += " ORDER BY rowid"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)

        rows = conn.execute(sql, params).fetchall()

        for r in rows:
            docs.append(
                Document(
                    page_content=r["text"] or "",
                    metadata={
                        "chunk_id": r["chunk_id"],
                        "doc_id": r["doc_id"],
                        "source": "sqlite_chunks",
                    },
                )
            )
    return docs


def evaluate_rag(dataset):
    from ragas.metrics import faithfulness, answer_relevancy
    from ragas import evaluate

    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    print(result)


if __name__ == "__main__":
    docs = load_docs_from_sqlite(config.sqlite.path)
    docs = [d for d in docs if len(d.page_content) > 50]
    print("Chunks:", len(docs), "Filtered docs:", len(docs))

    TEMP_MODEL = ChatOpenAI(
        api_key=config.embed.token,  # type: ignore
        base_url="https://api.siliconflow.cn/v1",
        model="deepseek-ai/DeepSeek-V3.2",
        temperature=0,
    )

    run_config = RunConfig(max_workers=1, timeout=180)
    generator = TestsetGenerator.from_langchain(
        llm=TEMP_MODEL, embedding_model=EMBED_MODEL
    )
    testset = generator.generate_with_langchain_docs(
        docs,
        testset_size=2,
        run_config=run_config,
        raise_exceptions=False,
        with_debugging_logs=True,
    )
    print(testset)

    df = testset.to_pandas()  # type: ignore
    print(df.head())

    out_path = "data/rag_testset.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
