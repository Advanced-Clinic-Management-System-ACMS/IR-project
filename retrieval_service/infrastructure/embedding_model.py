"""Pre-loaded SentenceTransformer encoder (loaded at service startup, not first query)."""
from __future__ import annotations

import numpy as np

_MODEL_CACHE: dict[str, object] = {}


def preload_model(model_name: str) -> None:
    from sentence_transformers import SentenceTransformer

    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)


def encode_text(text: str, model_name: str) -> np.ndarray:
    if model_name not in _MODEL_CACHE:
        preload_model(model_name)
    model = _MODEL_CACHE[model_name]
    vector = model.encode(text, normalize_embeddings=True)
    return np.asarray(vector, dtype=np.float32)
