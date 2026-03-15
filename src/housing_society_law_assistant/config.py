import os
from pathlib import Path

from dotenv import load_dotenv

APP_TITLE = "Housing Society Law Assistant"
# Resolve project root in an OS-independent way
BASE_DIR = Path(__file__).resolve().parent.parent.parent

load_dotenv(os.path.join(BASE_DIR, ".env"))
# Use pathlib for OS-independent paths and ensure directories exist
DATA_DIR = BASE_DIR.joinpath("data")
UPLOAD_DIR = DATA_DIR.joinpath("uploads")
VECTOR_DB_DIR = BASE_DIR.joinpath("vector_db")
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "astra").strip().lower()
COLLECTION_NAME = "maha_society"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_KEY = os.getenv("GEMINI_KEY", "").strip()
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT", "").strip()
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN", "").strip()
ASTRA_DB_NAMESPACE = os.getenv("ASTRA_DB_NAMESPACE", "").strip() or None
ASTRA_DB_ENVIRONMENT = os.getenv("ASTRA_DB_ENVIRONMENT", "").strip() or None
