from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB_NAME = "ir_project"
MONGO_DOCS_COLLECTION = "documents"

SERVICE_PORTS = {
    "preprocessing": 8001,
    "indexing": 8002,
    "retrieval": 8003,
    "query_refinement": 8004,
    "evaluation": 8005,
}

SERVICE_URLS = {
    name: f"http://127.0.0.1:{port}"
    for name, port in SERVICE_PORTS.items()
}

DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75
DEFAULT_TOP_K = 10
