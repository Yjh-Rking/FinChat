import uuid
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models.types import Chunk


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


def split_text(text, doc_id, chunk_size, chunk_overlap):
    splitter = get_splitter(chunk_size, chunk_overlap)
    texts = splitter.split_text(text)
    chunks = []
    for t in texts:
        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                text=t,
                metadata={},
            )
        )
    return chunks
