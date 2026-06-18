"""
Core query reformulation: spelling correction (edit distance) and synonym expansion (WordNet).
"""
from __future__ import annotations

import re

from nltk.corpus import wordnet


class QueryRefinerCore:
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

    def correct_spelling(self, tokens: list[str]) -> list[str]:
        corrected: list[str] = []
        for token in tokens:
            lower = token.lower()
            if lower in self._common_typos:
                corrected.append(self._common_typos[lower])
                continue
            if re.fullmatch(r"[a-zA-Z]+", token) and len(token) > 4:
                suggestions = wordnet.synsets(lower)
                if not suggestions and self._looks_like_typo(lower):
                    replacement = self._closest_wordnet_match(lower)
                    corrected.append(replacement or token)
                    continue
            corrected.append(token)
        return corrected

    @staticmethod
    def _looks_like_typo(word: str) -> bool:
        return any(wordnet.synsets(candidate) for candidate in (word[:-1], word + "e", word.replace("ie", "ei")))

    @staticmethod
    def _closest_wordnet_match(word: str) -> str | None:
        for candidate in {word[:-1], word + "e", word.replace("ie", "ei"), word.replace("ei", "ie")}:
            if wordnet.synsets(candidate):
                return candidate
        return None

    def expand_synonyms(self, tokens: list[str]) -> list[str]:
        expanded: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            lower = token.lower()
            if lower not in seen:
                expanded.append(token)
                seen.add(lower)

            for syn in wordnet.synsets(lower):
                for lemma in syn.lemmas()[:1]:
                    synonym = lemma.name().replace("_", " ")
                    if synonym.lower() not in seen:
                        expanded.append(synonym)
                        seen.add(synonym.lower())
                break
        return expanded
