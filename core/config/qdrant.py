from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="qdrant_",
        extra="ignore",
        case_sensitive=False,
    )
    path: str = "data/qdrant.db"
    url: str = "http://localhost:6333"
    collection: str = "chunks"
