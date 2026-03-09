from pydantic_settings import BaseSettings, SettingsConfigDict


class SQLiteSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="sqlite_",
        extra="ignore",
        case_sensitive=False,
    )
    path: str = "data/rag.db"
