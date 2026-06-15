# preprocessing_service/processor.py
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
        """تحميل موارد NLTK إذا لم تكن موجودة"""
        resources = ["punkt", "punkt_tab", "stopwords"]
        for resource in resources:
            try:
                if "punkt" in resource:
                    nltk.data.find(f"tokenizers/{resource}")
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

    def stem_tokens(self, tokens: list[str]) -> list[str]:
        """تحويل الكلمات إلى جذورها (Stemming)"""
        return [self.stemmer.stem(token) for token in tokens]

    def process_text(self, text: str) -> list[str]:
        """معالجة نص كامل: تطبيع → توكنايز → stemming"""
        tokens = self.tokenize(text)
        return self.stem_tokens(tokens)

    def combine_document_fields(self, document: DocumentInput) -> str:
        """
        دمج جميع حقول الوثيقة (Title, Author, Metadata, Text) في نص واحد للتنظيف.
        يتم تكرار العنوان مرتين لإعطائه وزناً أكبر في البحث.
        """
        parts = []
        
        # 1. العنوان (نكرره مرتين لأهميته)
        if document.title and document.title.strip():
            parts.append(document.title)
            parts.append(document.title)  # تكرار
        
        # 2. المؤلف (إذا وجد)
        if hasattr(document, 'author') and document.author and document.author.strip():
            parts.append(document.author)
        
        # 3. أي Metadata إضافي (إذا وجد)
        if hasattr(document, 'metadata') and document.metadata:
            for key, value in document.metadata.items():
                if isinstance(value, str) and value.strip():
                    parts.append(value)
                elif isinstance(value, list):
                    for v in value:
                        if isinstance(v, str) and v.strip():
                            parts.append(v)
        
        # 4. النص الأساسي
        if document.text and document.text.strip():
            parts.append(document.text)
        
        # دمج الكل بمسافات
        return " ".join(parts)

    def process_document(self, document: DocumentInput) -> ProcessedDocument:
        """
        معالجة وثيقة كاملة: دمج الحقول → تنظيف → توكنايز → stemming
        """
        # دمج جميع الحقول في نص واحد
        combined_text = self.combine_document_fields(document)
        
        # معالجة النص المدمج
        tokens = self.process_text(combined_text)
        
        # إرجاع النتيجة مع الاحتفاظ بالنص الأصلي للعرض
        return ProcessedDocument(
            doc_id=document.doc_id,
            tokens=tokens,
            original_text=document.text or "",
            title=document.title,  # نحتفظ بالعنوان منفصلاً للعرض
        )