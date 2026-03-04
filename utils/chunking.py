import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 / Qwen 等常用编码
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", "。", " ", ""],
    )
