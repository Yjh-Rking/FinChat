from ragas.testset import TestsetGenerator
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
from .config import config, EMBED_MODEL
from .sqlite import load_docs_from_sqlite


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
