"""
This file contains the core Natural Language Processing (NLP) logic.
It handles NLTK initialization, text normalization, tokenization, and stemming.
It is isolated from APIs, external dependencies, and data schemas.
"""
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from shared.config import NLP_NORMALIZATION_MODE


class NLPEngine:
    def __init__(self, language: str = "english") -> None:
        self._ensure_nltk_resources()
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(language))
        self.normalization_mode = NLP_NORMALIZATION_MODE

    @staticmethod
    def _ensure_nltk_resources() -> None:
        """تحميل موارد NLTK إذا لم تكن موجودة"""
        resources = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
        for resource in resources:
            try:
                if "punkt" in resource:
                    nltk.data.find(f"tokenizers/{resource}")
                elif resource in {"wordnet", "omw-1.4"}:
                    nltk.data.find(f"corpora/{resource}")
                else:
                    nltk.data.find(f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

    def normalize(self, text: str) -> str:
        """تطبيع النص: أحرف صغيرة، إزالة روابط، أرقام، رموز"""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", " ", text)  # إزالة الروابط
        text = re.sub(r"\d+", " ", text)  # إزالة الأرقام
        text = text.translate(str.maketrans("", "", string.punctuation))  # إزالة الرموز
        text = re.sub(r"\s+", " ", text).strip()  # إزالة المسافات الزائدة
        return text

    def tokenize(self, text: str) -> list[str]:
        """تقسيم النص إلى كلمات وإزالة كلمات التوقف"""
        normalized = self.normalize(text)
        tokens = word_tokenize(normalized)
        # إزالة الكلمات القصيرة وكلمات التوقف
        return [
            token for token in tokens 
            if token and len(token) > 1 and token not in self.stop_words
        ]

    def lemmatize_tokens(self, tokens: list[str]) -> list[str]:
        """تحويل الكلمات إلى lemmas (Lemmatization)"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def stem_tokens(self, tokens: list[str]) -> list[str]:
        """تحويل الكلمات إلى جذورها (Stemming)"""
        return [self.stemmer.stem(token) for token in tokens]

    def normalize_tokens(self, tokens: list[str]) -> list[str]:
        """Stemming (افتراضي للفهرس الحالي) أو Lemmatization عند تفعيلها."""
        if self.normalization_mode == "lemma":
            return self.lemmatize_tokens(tokens)
        return self.stem_tokens(tokens)

    def process_text(self, text: str) -> list[str]:
        """معالجة نص كامل: تطبيع → توكنايز → stemming/lemmatization"""
        tokens = self.tokenize(text)
        return self.normalize_tokens(tokens)