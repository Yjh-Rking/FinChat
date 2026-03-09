from pydantic_settings import BaseSettings, SettingsConfigDict


class TavilySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="tavily_",
        extra="ignore",
        case_sensitive=False,
    )
    url: str = "tavily_url"
    token: str = "tavily_token"
