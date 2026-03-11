# config/__init__.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from config.logging import LoggingSettings
from config.sqlite import SQLiteSettings
from config.qdrant import QdrantSettings
from config.tavily import TavilySettings
from config.openai import OpenAISettings


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Other
    data_dir: str = "./data/"

    # Settings
    log: LoggingSettings = LoggingSettings()
    sqlite: SQLiteSettings = SQLiteSettings()
    qdrant: QdrantSettings = QdrantSettings()
    tavily: TavilySettings = TavilySettings()
    openai: OpenAISettings = OpenAISettings()


config = Config()
config.log.setup_logging()

__all__ = ["config"]
