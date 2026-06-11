import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from shared.schemas import DocumentInput, ProcessedDocument


class TextPreprocessor:
    def __init__(self, language: str = "english") -> None:
        self._ensure_nltk_resources()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words(language))

    @staticmethod
    def _ensure_nltk_resources() -> None:
        resources = ("punkt", "punkt_tab", "stopwords")
        for resource in resources:
            try:
                nltk.data.find(
                    f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}"
                )
            except LookupError:
                nltk.download(resource, quiet=True)

    def normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"\d+", " ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> list[str]:
        normalized = self.normalize(text)
        tokens = word_tokenize(normalized)
        return [token for token in tokens if token and token not in self.stop_words]

    def stem_tokens(self, tokens: list[str]) -> list[str]:
        return [self.stemmer.stem(token) for token in tokens]

    def process_text(self, text: str) -> list[str]:
        tokens = self.tokenize(text)
        return self.stem_tokens(tokens)

    def process_document(self, document: DocumentInput) -> ProcessedDocument:
        content = document.text
        if document.title:
            content = f"{document.title} {document.text}"

        return ProcessedDocument(
            doc_id=document.doc_id,
            tokens=self.process_text(content),
            original_text=document.text,
        )
