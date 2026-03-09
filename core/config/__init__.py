from pydantic import BaseModel
from openai import OpenAI

from core.config.logging import LoggingSettings
from core.config.data import DataSettings
from core.config.embed import EmbedSettings
from core.config.chat import ChatSettings
from core.config.sqlite import SQLiteSettings
from core.config.qdrant import QdrantSettings
from core.config.tavily import TavilySettings


class Config(BaseModel):
    # Settings
    embed: EmbedSettings = EmbedSettings()
    chat: ChatSettings = ChatSettings()
    sqlite: SQLiteSettings = SQLiteSettings()
    qdrant: QdrantSettings = QdrantSettings()
    tavily: TavilySettings = TavilySettings()
    log: LoggingSettings = LoggingSettings()
    data: DataSettings = DataSettings()


def create_embedding(text: str) -> list[float]:
    """Create embedding using OpenAI's embeddings API."""
    client = OpenAI(
        api_key=config.embed.token,
        base_url="https://api.siliconflow.cn/v1/",
    )
    response = client.embeddings.create(
        model=config.embed.model,
        input=text,
    )
    return response.data[0].embedding


def create_chat(prompt: str, system_message: str = "") -> str:
    """Chat using OpenAI's chat completions API."""
    client = OpenAI(
        api_key=config.chat.token,
        base_url=config.chat.url,
    )
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=config.chat.model,
        messages=messages,
        temperature=0,
    )
    content = response.choices[0].message.content  # type: ignore
    # Strip thinking tags if present
    if "<think>" in content:
        content = content.split("</think>")[-1].strip()
    return content


config = Config()
config.log.setup_logging()
EMBED_MODEL = create_embedding
CHAT_MODEL = create_chat

__all__ = [
    "LoggingSettings",
    "DataSettings",
    "EmbedSettings",
    "ChatSettings",
    "SQLiteSettings",
    "QdrantSettings",
    "TavilySettings",
    "Config",
    "config",
    "EMBED_MODEL",
    "CHAT_MODEL",
]
