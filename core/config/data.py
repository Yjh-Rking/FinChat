from pydantic_settings import BaseSettings, SettingsConfigDict


class DataSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="data_",
        extra="ignore",
        case_sensitive=False,
    )
    path: str = "data/md/"
