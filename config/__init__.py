# config/__init__.py

from pydantic import BaseModel
from config.logging import LoggingSettings
from config.data import DataSettings
from config.sqlite import SQLiteSettings
from config.qdrant import QdrantSettings
from config.tavily import TavilySettings
from config.openai import OpenAISettings


class Config(BaseModel):
    # Settings
    log: LoggingSettings = LoggingSettings()
    sqlite: SQLiteSettings = SQLiteSettings()
    qdrant: QdrantSettings = QdrantSettings()
    tavily: TavilySettings = TavilySettings()
    data: DataSettings = DataSettings()
    openai: OpenAISettings = OpenAISettings()


config = Config()
config.log.setup_logging()

__all__ = ["config"]
