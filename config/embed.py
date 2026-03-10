from pydantic_settings import BaseSettings, SettingsConfigDict


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
