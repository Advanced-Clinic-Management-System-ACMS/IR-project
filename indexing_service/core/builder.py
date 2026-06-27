"""
Core indexing logic using industry-standard IR libraries:
- sklearn.feature_extraction.text.TfidfVectorizer
- rank_bm25.BM25Okapi
- sentence-transformers + FAISS (offline vector store)
"""
from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from shared.config import DEFAULT_EMBEDDING_MODEL
from shared.schemas import ProcessedDocument


class IndexBuilderCore:
    @staticmethod
    def build_sklearn_tfidf(
        documents: list[ProcessedDocument],
    ) -> tuple[TfidfVectorizer, object, list[str]]:
            # 1. تحويل الوثائق إلى نصوص (ضم الكلمات بمسافات)

        corpus = [" ".join(doc.tokens) for doc in documents]
                 # 2. استخراج معرفات الوثائق

        doc_ids = [doc.doc_id for doc in documents]
            # 3. إنشاء مدرب TF-IDF#TfidfVectorizer   مدرب جاهز للاستخدام
        vectorizer = TfidfVectorizer(norm="l2", use_idf=True, sublinear_tf=True)#norm="l2"يوحد المتجهات (يجعل طولها 1)#use_idf=Trueيستخدم IDF (ندرة الكلمة)#sublinear_tf=Trueيطبق تشبع على TF (log(1+TF))
            # 4. تدريب النموذج وتحويل الوثائق إلى متجهات
        matrix = vectorizer.fit_transform(corpus)
            # 5. إرجاع النتائج
        return vectorizer, matrix, doc_ids

    @staticmethod
    def build_rank_bm25(
        documents: list[ProcessedDocument],
    ) -> tuple[BM25Okapi, list[str]]:
        # 1. استخراج الكلمات (tokens) من كل وثيقة
        tokenized_corpus = [doc.tokens for doc in documents]
            # 2. استخراج معرفات الوثائق
        doc_ids = [doc.doc_id for doc in documents]
            # 3. إنشاء نموذج BM25
        bm25 = BM25Okapi(tokenized_corpus)
            # 4. إرجاع النموذج والمعرفات
        return bm25, doc_ids
        #ماذا يفعل BM25Okapi خلف الكواليس؟
        # يحسب IDF لكل كلمة

        # يحسب طول كل وثيقة

        # يحسب متوسط طول الوثائق

        # يخزن كل هذه الإحصائيات للاستخدام في البحث

    @staticmethod
    def build_sentence_embeddings(
        documents: list[ProcessedDocument],
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 256,
    ) -> tuple[np.ndarray, str]:
        from sentence_transformers import SentenceTransformer
#      # 1. تحميل نموذج Embedding
        model = SentenceTransformer(model_name)
            # 2. استخراج النصوص من الوثائق
        texts = [doc.original_text or " ".join(doc.tokens) for doc in documents]
            # 3. تحويل النصوص إلى متجهات (Embeddings)
        embeddings = model.encode(
            texts,
            batch_size=batch_size, # عدد الوثائق في كل دفعة
            show_progress_bar=len(documents) > 1000, # عرض شريط التقدم
            normalize_embeddings=True,# توحيد المتجهات
        )
        return np.asarray(embeddings, dtype=np.float32), model_name
# مكتبة من Facebook للبحث السريع عن المتجهات
#تستخدم في أنظمة البحث الدلالي (Semantic Search)
#أسرع من البحث العادي بمئات المرات
#يحسب تشابه المتجهات باستخدام الضرب النقطي (Inner Product)
#يرتب النتائج من الأعلى تشابهاً إلى الأقل
    @staticmethod
    def build_faiss_index(embeddings: np.ndarray):
        import faiss
    # 1. معرفة طول المتجه
        dimension = embeddings.shape[1]
       # ينشئ فهرس للبحث عن المتجهات الأقرب
        index = faiss.IndexFlatIP(dimension)
            # 3. إضافة المتجهات إلى الفهرس للبحث السريع
        index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        return index

    @staticmethod
    def summary_stats(documents: list[ProcessedDocument]) -> dict:
        total = len(documents)#عدد الوثائق 
        avg_len = sum(len(doc.tokens) for doc in documents) / (total or 1)#مجموع أطوال الوثائق
        return {
            "document_count": total,
            "avg_doc_length": round(avg_len, 2),
            "vocabulary_size": len({token for doc in documents for token in doc.tokens}),#عدد الكلمات الفريدة
        }
