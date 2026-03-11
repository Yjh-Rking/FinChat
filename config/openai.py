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
    embed: str = "embed model name"
    chat_url: str = ""
    chat_api: str = ""
    chat: str = "chat model name"
    rerank: str = "rerank model name"
