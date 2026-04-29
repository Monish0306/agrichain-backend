from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/agrichain"
    secret_key: str = "change-me"

    openweather_api_key: str | None = None
    groq_api_key: str | None = None

    monitor_username: str | None = None
    monitor_password: str | None = None

    redis_url: str | None = None

    jwt_algorithm: str = "HS256"
    farmer_access_token_hours: int = 24
    merchant_access_token_hours: int = 12
    monitor_access_token_hours: int = 2


settings = Settings()

