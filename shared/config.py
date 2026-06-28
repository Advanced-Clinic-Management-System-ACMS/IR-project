from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"

# Storage:
#   data/documents.db          — SQLite original documents (online retrieval)
#   data/raw/<dataset>.jsonl   — offline pipeline backup
#   data/processed/<dataset>.jsonl
#   data/indexes/<dataset>/    — compressed sklearn/rank_bm25/faiss artifacts

DEFAULT_IR_DATASET = "lotte/lifestyle/dev/forum"
DEFAULT_DATASET_NAME = "lotte_lifestyle_dev_forum"

# Fixed session history used in personalization evaluation (Requirement #16).
DEFAULT_EVAL_HISTORY = [
    "cat health food nutrition",
    "vet visit kitten symptoms",
    "pet grooming advice forum",
]

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

# "stem" keeps compatibility with the built LoTTE index; use "lemma" only after re-indexing.
NLP_NORMALIZATION_MODE = os.environ.get("IR_NLP_MODE", "stem")
