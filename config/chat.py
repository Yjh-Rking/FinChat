from pydantic_settings import BaseSettings, SettingsConfigDict


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
