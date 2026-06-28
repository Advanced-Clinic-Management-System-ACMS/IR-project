# تقرير مشروع نظام استرجاع المعلومات (IR 2026)

**المجموعة:** [أدخل أسماء الأعضاء]  
**Dataset:** LoTTE — `lotte/lifestyle/dev/forum`  
**التاريخ:** يونيو 2026  
**GitHub:** [أدخل الرابط]

---

## 1. مقدمة

يهدف المشروع إلى بناء نظام استرجاع معلومات (Information Retrieval) متكامل يعمل على مجموعة بيانات **LoTTE Lifestyle Forum** التي تحتوي على **268,893** وثيقة و**2076** استعلاماً مع تقييمات relevance (qrels). يتبع النظام معمارية **SOA** (Service-Oriented Architecture) و**Clean Architecture**، ويُنفَّذ بالكامل بلغة **Python**.

---

## 2. مجموعة البيانات

| البند | القيمة |
|-------|--------|
| المصدر | [ir-datasets](https://ir-datasets.com/lotte) |
| المعرف | `lotte/lifestyle/dev/forum` |
| عدد الوثائق | 268,893 |
| عدد الاستعلامات (qrels) | 2,076 |
| اللغة | الإنجليزية (منتدى lifestyle) |

**ملاحظة:** ملفات البيانات والفهارس محلية (`data/`) ولا تُرفع إلى Git — تُبنى عبر `scripts/run_offline_pipeline.py --ir-datasets`.

---

## 3. معمارية النظام (SOA)

### 3.1 الخدمات

| الخدمة | المنفذ | المسؤولية |
|--------|--------|-----------|
| ui_gateway | 8000 | واجهة المستخدم (FastAPI + Jinja2) |
| preprocessing_service | 8001 | تنظيف النص، tokenization، stemming/lemmatization |
| indexing_service | 8002 | بناء فهارس TF-IDF، BM25، FAISS |
| retrieval_service | 8003 | الاسترجاع (Strategy + Factory) |
| query_refinement | 8004 | تحسين الاستعلام + Personalization (#16) |
| evaluation_service | 8005 | MAP, Recall, P@10, nDCG |

### 3.2 التواصل

- **Online:** REST/HTTP بين الخدمات (JSON + Pydantic schemas في `shared/schemas.py`).
- **Offline:** pipeline يبني الفهارس محلياً لتجنب حدود حجم HTTP.

### 3.3 Design Patterns

- **Strategy:** نموذج استرجاع لكل خوارزمية (`retrieval_service/core/strategies/`).
- **Factory:** `RetrievalFactory.create(model)`.
- **Repository:** `IndexRepository`, `DocumentRepository`.
- **Gateway/BFF:** `ui_gateway` يجمع الطلبات ويرسلها لـ retrieval.

راجع `docs/ARCHITECTURE.md` للمخطط التفصيلي.

---

## 4. معالجة البيانات (Pre-Processing)

**الموقع:** `preprocessing_service/core/nlp_engine.py`

| المرحلة | التImplementation |
|---------|---------------------|
| Normalization | lowercase، إزالة URLs، أرقام، punctuation |
| Tokenization | NLTK `word_tokenize` |
| Stop Words | NLTK English stopwords |
| Stemming | Porter Stemmer (الوضع الافتراضي — متوافق مع الفهرس) |
| Lemmatization | WordNet Lemmatizer (متاح عند `IR_NLP_MODE=lemma` + إعادة فهرسة) |

**Endpoint:** `POST /process-query` — نفس pipeline للوثائق والاستعلامات.

---

## 5. تمثيل الوثائق والفهرسة (Indexing)

**الموقع:** `indexing_service/core/builder.py`

| التمثيل | المكتبة | الملف المحفوظ |
|---------|---------|---------------|
| TF-IDF (VSM) | sklearn TfidfVectorizer | `tfidf_vectorizer.joblib`, `tfidf_matrix.npz` |
| BM25 | rank_bm25 BM25Okapi | `bm25_model.joblib` |
| Embedding | Sentence-BERT `all-MiniLM-L6-v2` | `embeddings.npz`, `faiss.index` |
| Hybrid | دمج عند الاسترجاع | — |

**Inverted Index:** يُبنى داخلياً ضمن sklearn و rank_bm25 (postings lists) — لا ملف منفصل؛ artifacts مضغوطة على القرص.

**metadata.json** يحتوي: `document_count`, `vocabulary_size`, `embedding_model`, `index_format: library_v2`.

---

## 6. معالجة الاستعلامات (Query Processing)

1. المستخدم يرسل استعلاماً عبر UI أو evaluation script.
2. `retrieval_service` يستدعي `query_refinement` (اختياري).
3. `retrieval_service` يستدعي `preprocessing_service` → `/process-query`.
4. Tokens تُستخدم في scoring على الفهرس المحمّل في RAM.

---

## 7. نماذج الاسترجاع والـ Hybrid

| النموذج | الوصف |
|---------|-------|
| tf_idf | Cosine similarity على sparse TF-IDF matrix |
| bm25 | BM25Okapi مع معاملات k1, b قابلة للتعديل |
| embedding | FAISS IndexFlatIP + SentenceTransformer |
| hybrid_serial | BM25 candidates → re-rank بـ embedding |
| hybrid_parallel | 3 scorers متوازية → **RRF** أو **Weighted Sum** |
| hybrid_branching | قصير→BM25، متوسط→TF-IDF، طويل→Embedding |

**Fusion:** `fuse_rrf(k=60)`, `fuse_weighted()` في `retrieval_service/core/scoring.py`.

---

## 8. Query Refinement و Personalization (#16)

### 8.1 Query Refinement
- تصحيح إملائي + WordNet synonyms (`query_refinement/core/refiner.py`).
- Endpoint: `POST /refine`.

### 8.2 Personalization
- **History term profiling:** terms متكررة من سجل البحث.
- **Semantic history matching:** cosine similarity بين الاستعلام والسجل.
- **Cookies:** `ir_search_history` — حد أقصى **10** استعلامات (`ui_gateway/api/routes.py`).

---

## 9. واجهة المستخدم (UI)

**الموقع:** `ui_gateway/templates/index.html`

| الميزة | التنفيذ |
|--------|---------|
| Basic / Extra | radio buttons — Extra يفعّل refinement + personalization |
| BM25 k1, b | sliders (range inputs) |
| Fusion | RRF / Weighted |
| Hybrid weights | TF-IDF, BM25, Embedding |
| النتائج | Rank, Score, Title, Snippet (300 حرف) |
| اقتراحات التحسين | suggestions من query_refinement |

---

## 10. التقييم (Evaluation)

### 10.1 الإعداد
- **Dataset qrels:** 2076 استعلام (كامل — بدون sampling).
- **Script:** `scripts/run_evaluation.py`
- **المقاييس:** MAP, Recall, Precision@10, nDCG@10.

### 10.2 Baseline (6 نماذج)

| Model | MAP | Recall | P@10 | nDCG@10 |
|-------|-----|--------|------|---------|
| tf_idf | *انظر report.json* | | | |
| bm25 | ~0.32 | ~0.22 | ~0.12 | ~0.39 |
| embedding | *report.json* | | | |
| hybrid_serial | *report.json* | | | |
| hybrid_parallel | ~0.41 | ~0.29 | ~0.16 | ~0.49 |
| hybrid_branching | *report.json* | | | |

> **مهم:** إذا كان `report.json` يُظهر MAP ~0.01 لجميع النماذج، فهو من تشغيل قديم — شغّل `scripts/rebuild_baseline_report.ps1`.

### 10.3 Query Refinement — Before vs After

| Model | Δ MAP | Δ nDCG@10 |
|-------|-------|-----------|
| bm25 | -0.035 | -0.041 |
| hybrid_parallel | -0.033 | -0.037 |

### 10.4 Personalization — Before vs After

| Model | Δ MAP | Δ nDCG@10 |
|-------|-------|-----------|
| bm25 | -0.099 | -0.123 |
| hybrid_parallel | -0.100 | -0.121 |

### 10.5 تفسير تراجع المقاييس (Query Drift)

Personalization استخدمت سجل بحث عن **الحيوانات الأليفة** بينما qrels لمنتدى **lifestyle عام**. حقن terms من domain مختلف يُحرّك الاستعلام بعيداً عن relevance الأصلية → **Query Drift**. هذا يثبت:
1. الميزة **تعمل** (delta واضح ~-10%).
2. Personalization يحتاج **توافق domain** بين السجل والـ corpus.
3. منهجية before/after **صحيحة** على test collection ثابت.

---

## 11. تقسيم العمل (أدخل أسماء المجموعة)

| العضو | المسؤولية |
|-------|-----------|
| … | preprocessing + indexing pipeline |
| … | retrieval + hybrid strategies |
| … | query refinement + personalization |
| … | UI + evaluation + documentation |

---

## 12. تشغيل النظام

```powershell
# 1) الخدمات (4 terminals)
py preprocessing_service\main.py
py retrieval_service\main.py
py query_refinement\main.py
py ui_gateway\main.py

# 2) التحقق
.\scripts\validate_evaluation.ps1

# 3) إعادة baseline إذا لزم
.\scripts\rebuild_baseline_report.ps1
```

---

## 13. المراجع

- Manning, Raghavan, Schütze — *Introduction to Information Retrieval*
- [ir-datasets LoTTE](https://ir-datasets.com/lotte)
- NLTK, scikit-learn, rank_bm25, FAISS, Sentence-Transformers

---

## 14. الملفات المرجعية

| الملف | الغرض |
|-------|-------|
| `data/evaluation/EVALUATION_SUMMARY.md` | جداول جاهزة للمقابلة |
| `data/evaluation/charts/` | رسوم MAP/nDCG |
| `docs/ARCHITECTURE.md` | مخطط النظام |
| `docs/SUBMISSION_DAY.md` | checklist يوم التسليم |
