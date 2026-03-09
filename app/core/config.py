from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # Google Gemini Settings
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "models/gemini-2.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "models/gemini-embedding-001"

    # OpenAI Settings (keeping for fallback or if needed)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Vector Store Settings
    VECTOR_STORE_TYPE: str = "chroma"  # "chroma" or "pinecone"
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma_db"
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    PINECONE_INDEX_NAME: Optional[str] = None

    # RAG Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.7

    # API Settings
    PROJECT_NAME: str = "Production RAG System"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

settings = Settings()
