from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application Configuration
    """
    PROJECT_NAME: str = "IES_EV_System"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    
    # Backend
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LOG_LEVEL: str = "info"
    
    # PostgreSQL
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: str = "5432"
    
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # Redis
    REDIS_HOST: str
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # DeepSeek AI
    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_IES_API_KEY: Optional[str] = None
    DEEPSEEK_MODEL: str = "deepseek-coder"

    # Groq AI (fast inference)
    GROQ_API_KEY: Optional[str] = None
    GROQ_API_URL: str = "https://api.groq.com/openai/v1"

    # Gemini AI
    GEMINI_API_KEY: Optional[str] = None

    # Sentry error tracking
    SENTRY_DSN: Optional[str] = None

    # External APIs
    ORS_API_KEY: Optional[str] = None
    OPENWEATHER_API_KEY: Optional[str] = None
    OPENCHARGE_API_KEY: Optional[str] = None
    
    model_config = SettingsConfigDict(
        env_file=[
            str(Path(__file__).resolve().parents[2] / ".env"),
            ".env"
        ],
        case_sensitive=True, 
        extra="ignore"
    )
settings = Settings()
