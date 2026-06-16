"""
This file acts as the Application layer orchestration.
It extracts data from the Pydantic input models, uses the NLP Engine to process the text,
and constructs the strongly-typed output models.
"""
from shared.schemas import DocumentInput, ProcessedDocument
from preprocessing_service.core.nlp_engine import NLPEngine

class DocumentProcessorService:
    def __init__(self) -> None:
        self.nlp_engine = NLPEngine()

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
        
        return " ".join(parts)

    def process_document(self, document: DocumentInput) -> ProcessedDocument:
        """
        معالجة وثيقة كاملة: دمج الحقول → تنظيف → توكنايز → stemming
        """
        combined_text = self.combine_document_fields(document)
        tokens = self.nlp_engine.process_text(combined_text)
        
        return ProcessedDocument(
            doc_id=document.doc_id,
            tokens=tokens,
            original_text=document.text or "",
            title=document.title, 
        )

    def process_raw_query(self, query_text: str) -> list[str]:
        """معالجة استعلام بحث مباشر"""
        return self.nlp_engine.process_text(query_text)