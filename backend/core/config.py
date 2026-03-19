import os
from pathlib import Path

from dotenv import load_dotenv


ENV_PATH = Path(__file__).resolve().parents[2] / "configs" / "dev.env"
load_dotenv(ENV_PATH)

POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")

FRED_API_KEY = os.getenv("FRED_API_KEY")
