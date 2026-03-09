import logging.config
from pathlib import Path
from typing import Literal
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class LoggingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="log_",
        extra="ignore",
        case_sensitive=False,
    )

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    file: Path = Path("logs/rag.log")
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def setup_logging(self):
        self.file.parent.mkdir(parents=True, exist_ok=True)

        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": self.format,
                },
            },
            "handlers": {
                "default": {
                    "level": self.level,
                    "formatter": "standard",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "level": self.level,
                    "formatter": "standard",
                    "class": "logging.FileHandler",
                    "filename": str(self.file),
                    "mode": "a",
                },
            },
            "loggers": {
                "": {
                    "handlers": ["default", "file"],
                    "level": self.level,
                    "propagate": False,
                },
                "httpx": {
                    "level": "WARNING",
                    "propagate": False,
                },
                "httpcore": {
                    "level": "WARNING",
                    "propagate": False,
                },
            },
        }

        logging.config.dictConfig(log_config)


class DataSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="data_",
        extra="ignore",
        case_sensitive=False,
    )
    path: str = "data/md/"


class EmbedSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="embed_",
        extra="ignore",
        case_sensitive=False,
    )
    url: str = "embed_url"
    token: str = "embed_token"
    model: str = "embed_model"


class ChatSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="chat_",
        extra="ignore",
        case_sensitive=False,
    )
    url: str = "chat_url"
    token: str = "chat_token"
    model: str = "chat_model"


class SQLiteSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="sqlite_",
        extra="ignore",
        case_sensitive=False,
    )
    path: str = "data/rag.db"


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


class Config(BaseModel):
    # Settings
    embed: EmbedSettings = EmbedSettings()
    chat: ChatSettings = ChatSettings()
    sqlite: SQLiteSettings = SQLiteSettings()
    qdrant: QdrantSettings = QdrantSettings()
    log: LoggingSettings = LoggingSettings()
    data: DataSettings = DataSettings()


config = Config()
config.log.setup_logging()
EMBED_MODEL = OpenAIEmbeddings(
    api_key=config.embed.token,  # type: ignore
    base_url="https://api.siliconflow.cn/v1/",
    model=config.embed.model,
)
CHAT_MODEL = ChatOpenAI(
    api_key=config.chat.token,  # type: ignore
    base_url=config.chat.url,
    model=config.chat.model,
    temperature=0,
    extra_body={"reasoning_split": True},
)
