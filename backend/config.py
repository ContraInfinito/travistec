# backend/config.py
from pydantic import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    AZURE_FACE_KEY: Optional[str] = None
    AZURE_FACE_ENDPOINT: Optional[str] = None
    PORT: int = 8000
    ENV: str = "development"
    # Orígenes permitidos separados por coma (opcional)
    FRONTEND_ORIGINS: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
