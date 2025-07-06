import logging
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application configuration read from environment or .env file"""
    GEMINI_API_KEY: str = "AIzaSyA5_SCGpT0kq6yoFLTBBt8j8PtgalL4JPM"
    GOOGLE_API_KEY: str = "AIzaSyA5_SCGpT0kq6yoFLTBBt8j8PtgalL4JPM"
    GOOGLE_CX: str = "e1c218b2f088a4a4e"
    MAX_RESULTS: int = 5
    MODEL: str = "gemini-2.0-flash"
    CACHE_FILE: str = "cache.json"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("deep_research_agent")
