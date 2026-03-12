from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="openai_",
        extra="ignore",
        case_sensitive=False,
    )

    embed_url: str = ""
    embed_api: str = ""
    embed_model: str = "embed model name"
    embed_size: int = 1024
    chat_url: str = ""
    chat_api: str = ""
    chat_model: str = "chat model name"
    rerank_url: str = ""
    rerank_api: str = ""
    rerank_model: str = "rerank model name"
