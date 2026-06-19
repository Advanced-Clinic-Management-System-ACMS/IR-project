"""
Core query reformulation: spelling correction and controlled synonym expansion (WordNet).
Designed as a lightweight, explainable refinement layer — no ML model required.
"""
from __future__ import annotations

import re

from nltk.corpus import wordnet


class QueryRefinerCore:
    MAX_SYNONYMS_TOTAL = 3

    def __init__(self) -> None:
        self._common_typos = {
            "teh": "the",
            "adn": "and",
            "taht": "that",
            "recieve": "receive",
            "informtion": "information",
            "artifical": "artificial",
            "retreival": "retrieval",
            "serach": "search",
            "retrival": "retrieval",
            "serch": "search",
            "qustion": "question",
            "recomend": "recommend",
            "recomendation": "recommendation",
            "definately": "definitely",
            "seperate": "separate",
            "occured": "occurred",
            "wht": "what",
            "whats": "what",
            "dont": "do not",
            "doesnt": "does not",
            "cant": "cannot",
        }
        self._ensure_nltk()

    @staticmethod
    def _ensure_nltk() -> None:
        import nltk

        for resource in ("wordnet", "omw-1.4"):
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

    @staticmethod
    def normalize_query(text: str) -> str:
        text = re.sub(r"\s+", " ", text.strip())
        return text

    def tokenize_query(self, query: str) -> list[str]:
        normalized = self.normalize_query(query)
        return normalized.split() if normalized else []

    def correct_spelling(self, tokens: list[str]) -> list[str]:
        corrected: list[str] = []
        for token in tokens:
            lower = token.lower()
            if lower in self._common_typos:
                corrected.append(self._common_typos[lower])
                continue
            if re.fullmatch(r"[a-zA-Z]+", token) and len(token) > 4:
                if not wordnet.synsets(lower) and self._looks_like_typo(lower):
                    replacement = self._closest_wordnet_match(lower)
                    corrected.append(replacement or token)
                    continue
            corrected.append(token)
        return corrected

    @staticmethod
    def _looks_like_typo(word: str) -> bool:
        candidates = (word[:-1], word + "e", word.replace("ie", "ei"), word.replace("ei", "ie"))
        return any(wordnet.synsets(candidate) for candidate in candidates)

    @staticmethod
    def _closest_wordnet_match(word: str) -> str | None:
        for candidate in {word[:-1], word + "e", word.replace("ie", "ei"), word.replace("ei", "ie")}:
            if wordnet.synsets(candidate):
                return candidate
        return None

    def expand_synonyms(self, tokens: list[str]) -> list[str]:
        expanded: list[str] = []
        seen: set[str] = set()
        added_synonyms = 0

        for token in tokens:
            lower = token.lower()
            if lower not in seen:
                expanded.append(token)
                seen.add(lower)

            if added_synonyms >= self.MAX_SYNONYMS_TOTAL:
                continue

            synsets = wordnet.synsets(lower)
            if not synsets:
                continue

            for lemma in synsets[0].lemmas()[:2]:
                synonym = lemma.name().replace("_", " ").lower()
                if synonym == lower or synonym in seen:
                    continue
                expanded.append(synonym)
                seen.add(synonym)
                added_synonyms += 1
                break

        return expanded

    def deduplicate_tokens(self, tokens: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(token)
        return result

    def refine(self, query: str) -> tuple[str, list[str]]:
        tokens = self.tokenize_query(query)
        corrected = self.correct_spelling(tokens)
        expanded = self.expand_synonyms(corrected)
        final_tokens = self.deduplicate_tokens(expanded)
        return " ".join(final_tokens), final_tokens
