from openai import OpenAI
from config import config


def embedding_model(text: str) -> list[float]:
    """Create embedding using OpenAI's embeddings API."""
    client = OpenAI(
        api_key=config.openai.embed_api,
        base_url=config.openai.embed_url,
    )
    response = client.embeddings.create(
        model=config.openai.embed,
        input=text,
    )
    return response.data[0].embedding


def chat_model(
    prompt: str, system_message: str = "", extra_body: dict = {"reasoning_split": True}
) -> str:
    """Chat using OpenAI's chat completions API."""
    client = OpenAI(
        api_key=config.openai.chat_api,
        base_url=config.openai.chat_url,
    )
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=config.openai.chat,
        messages=messages,
        temperature=0,
        extra_body=extra_body,
    )
    return response.choices[0].message.content  # type: ignore
