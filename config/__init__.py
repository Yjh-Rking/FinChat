# config/__init__.py

from pydantic import BaseModel
from config.logging import LoggingSettings
from config.data import DataSettings
from config.embed import EmbedSettings
from config.chat import ChatSettings
from config.sqlite import SQLiteSettings
from config.qdrant import QdrantSettings
from config.tavily import TavilySettings
from config.rerank import RerankSettings


class Config(BaseModel):
    # Settings
    embed: EmbedSettings = EmbedSettings()
    chat: ChatSettings = ChatSettings()
    sqlite: SQLiteSettings = SQLiteSettings()
    qdrant: QdrantSettings = QdrantSettings()
    tavily: TavilySettings = TavilySettings()
    log: LoggingSettings = LoggingSettings()
    data: DataSettings = DataSettings()
    rerank: RerankSettings = RerankSettings()


config = Config()
config.log.setup_logging()

__all__ = ["config"]
