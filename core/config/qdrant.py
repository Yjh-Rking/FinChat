from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="qdrant_",
        extra="ignore",
        case_sensitive=False,
    )
    host: str = "qdrant_host"
    port: int = 6333
    collection_name: str = "research_collection"
    path: str = "data/qdrant.db"
