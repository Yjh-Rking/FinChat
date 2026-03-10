from pydantic_settings import BaseSettings, SettingsConfigDict


class RerankSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="rerank_",
        extra="ignore",
        case_sensitive=False,
    )
    model: str = "BAAI/bge-reranker-v2-m3"
