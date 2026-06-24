"""
Personalization (Requirement 16): two IR techniques applied to search history.
1) History term profiling — expand query with frequent terms from past queries.
2) Semantic history matching — add terms from the most similar past query (embedding cosine).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from shared.config import DEFAULT_EMBEDDING_MODEL

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall", "can", "need", "what", "how",
    "why", "when", "where", "who", "which", "my", "your", "his", "her", "their", "our", "i",
}


_MODEL_CACHE: dict[str, object] = {}


def preload_model(model_name: str) -> None:
    from sentence_transformers import SentenceTransformer

    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)


def _encode_text(text: str, model_name: str) -> np.ndarray:
    preload_model(model_name)
    model = _MODEL_CACHE[model_name]
    vector = model.encode(text, normalize_embeddings=True)
    return np.asarray(vector, dtype=np.float32)


class PersonalizationEngine:
    @staticmethod
    def apply_history(
        query_tokens: list[str],
        history: list[str],
    ) -> tuple[list[str], list[str]]:
        if not history:
            return query_tokens, []

        expanded = list(query_tokens)
        applied: list[str] = []
        seen = {token.lower() for token in expanded}

        profile_terms = PersonalizationEngine._history_term_profile(history, max_terms=3)
        for term in profile_terms:
            if term.lower() not in seen:
                expanded.append(term)
                seen.add(term.lower())
        if profile_terms:
            applied.append("history_term_profile")

        semantic_terms = PersonalizationEngine._semantic_history_expansion(
            " ".join(query_tokens),
            history,
            max_terms=2,
        )
        for term in semantic_terms:
            if term.lower() not in seen:
                expanded.append(term)
                seen.add(term.lower())
        if semantic_terms:
            applied.append("semantic_history_similarity")

        return expanded, applied

    @staticmethod
    def _history_term_profile(history: list[str], max_terms: int) -> list[str]:
        counter: Counter[str] = Counter()
        for past_query in history:
            for token in past_query.lower().split():
                if len(token) > 2 and token not in _STOPWORDS:
                    counter[token] += 1
        return [term for term, _ in counter.most_common(max_terms)]

    @staticmethod
    def _semantic_history_expansion(
        query: str,
        history: list[str],
        max_terms: int,
    ) -> list[str]:
        preload_model(DEFAULT_EMBEDDING_MODEL)
        query_emb = _encode_text(query, DEFAULT_EMBEDDING_MODEL)
        best_query = None
        best_score = -1.0
        for past in history:
            past_emb = _encode_text(past, DEFAULT_EMBEDDING_MODEL)
            score = float(np.dot(query_emb, past_emb))
            if score > best_score:
                best_score = score
                best_query = past

        if best_query is None or best_score < 0.45:
            return []

        query_lower = query.lower()
        terms = [
            token
            for token in best_query.split()
            if len(token) > 2 and token.lower() not in query_lower and token.lower() not in _STOPWORDS
        ]
        return terms[:max_terms]
