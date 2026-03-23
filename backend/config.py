from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import quote_plus, unquote

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / "configs" / "dev.env"
load_dotenv(ENV_PATH)


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "NQ ALPHA Financial AI")
    app_version: str = os.getenv("APP_VERSION", "1.0.0")

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api")
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2-vision:11b-instruct-q4_K_M")
    ollama_advisor_model: str = os.getenv("OLLAMA_ADVISOR_MODEL", "llama3.2-vision:11b-instruct-q4_K_M")
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma")
    request_timeout_seconds: int = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "120"))
    newsdata_api_key: str = os.getenv("NEWSDATA_API_KEY", "")

    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "neuroquant")
    postgres_user: str = os.getenv("POSTGRES_USER", "postgres")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    database_url: str | None = os.getenv("DATABASE_URL")

    chroma_path: str = os.getenv(
        "CHROMA_PATH",
        str(PROJECT_ROOT / "backend" / "vectorstore" / "chroma_data"),
    )
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "financial_assistant_memory")
    memory_top_k: int = int(os.getenv("MEMORY_TOP_K", "5"))

    @property
    def postgres_url(self) -> str:
        if self.database_url:
            return self.database_url

        encoded_user = quote_plus(unquote(self.postgres_user))
        encoded_password = quote_plus(unquote(self.postgres_password))
        return (
            f"postgresql+psycopg2://{encoded_user}:{encoded_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
