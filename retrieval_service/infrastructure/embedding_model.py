"""Lazy-loaded sentence-transformer encoder for query-time semantic scoring."""
from __future__ import annotations

import numpy as np

_MODEL_CACHE: dict[str, object] = {}


def encode_text(text: str, model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    model = _MODEL_CACHE[model_name]
    vector = model.encode(text, normalize_embeddings=True)
    return np.asarray(vector, dtype=np.float32)
