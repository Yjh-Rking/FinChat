from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal
from pathlib import Path
import logging.config


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
