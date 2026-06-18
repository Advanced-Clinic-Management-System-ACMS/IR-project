from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"

# File-based storage (no MongoDB):
#   data/raw/<dataset>.json
#   data/processed/<dataset>.json
#   data/indexes/<dataset>/

DEFAULT_IR_DATASET = "lotte/lifestyle/dev/forum"
DEFAULT_DATASET_NAME = "lotte_lifestyle_dev_forum"

SERVICE_PORTS = {
    "preprocessing": 8001,
    "indexing": 8002,
    "retrieval": 8003,
    "query_refinement": 8004,
    "evaluation": 8005,
    "ui_gateway": 8000,
}

SERVICE_URLS = {
    name: f"http://127.0.0.1:{port}"
    for name, port in SERVICE_PORTS.items()
}

DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75
DEFAULT_TOP_K = 10
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
