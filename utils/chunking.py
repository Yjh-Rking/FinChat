import tiktoken
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_markdown(path):
    loader = DirectoryLoader(path, glob="**/*.md")
    return loader.load()


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 / Qwen 等常用编码
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def get_splitter(chunk_size, chunk_overlap):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", "。", " ", ""],
    )


def split_docs(chunk_size, chunk_overlap, docs):
    splitter = get_splitter(chunk_size, chunk_overlap)
    return splitter.split_documents(docs)
