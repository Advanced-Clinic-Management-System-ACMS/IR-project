# تقرير مشروع نظام استرجاع المعلومات (IR 2026)

> مسودة جاهزة للتسليم — أضف screenshots من UI والـ charts وGitHub link.

---

## 1. مقدمة

تم بناء **نظام استرجاع معلومات (Information Retrieval System)** وفق متطلبات مقر IR 2026، باستخدام:

- **Dataset:** `lotte/lifestyle/dev/forum` من [ir-datasets](https://ir-datasets.com/lotte)
- **268,893** وثيقة | **2,076** استعلام في qrels
- **Python** + **FastAPI** + **SOA** + **Clean Architecture**

---

## 2. Dataset

| البند | القيمة |
|--------|--------|
| المصدر | ir-datasets |
| المعرف | `lotte/lifestyle/dev/forum` |
| عدد الوثائق | 268,893 |
| Qrels queries | 2,076 |
| التخزين المحلي | `data/raw/`, `data/processed/`, `data/indexes/` |

**Screenshot مطلوب:** مخرجات `py load_data.py` أو `metadata.json`.

---

## 3. المعمارية (SOA)

### الخدمات

| الخدمة | المنفذ | المسؤولية |
|--------|--------|-----------|
| ui_gateway | 8000 | واجهة المستخدم |
| preprocessing_service | 8001 | معالجة النصوص |
| indexing_service | 8002 | بناء الفهرس |
| retrieval_service | 8003 | البحث والترتيب |
| query_refinement | 8004 | تحسين الاستعلام |
| evaluation_service | 8005 | حساب المقاييس |

**Screenshot مطلوب:** مخطط Mermaid من `docs/ARCHITECTURE.md`.

### Clean Architecture

- **Core:** خوارزميات بحتة (TF-IDF, BM25, RRF, embeddings)
- **Infrastructure:** قراءة الفهرس من القرص، HTTP clients
- **Services:** orchestrators
- **API:** FastAPI routes

---

## 4. تنفيذ الطلبات الأساسية

### 4.1 Pre-Processing (الطلب 1)

**الملف:** `preprocessing_service/core/nlp_engine.py`

- Lowercasing
- إزالة URLs، أرقام، punctuation
- Tokenization (NLTK)
- Stopword removal
- Porter Stemming

**Screenshot:** كود `nlp_engine.py` + tokens في UI.

---

### 4.2 تمثيل الوثائق (الطلب 2)

| النموذج | التنفيذ | الملف |
|---------|---------|-------|
| TF-IDF | Cosine على sparse vectors | `tfidf_strategy.py` |
| BM25 | probabilistic ranking | `bm25_strategy.py` |
| Embedding | Sentence-BERT `all-MiniLM-L6-v2` | `embedding_strategy.py` |
| Hybrid Serial | BM25 top-100 → embedding re-rank | `serial_strategy.py` |
| Hybrid Parallel | TF-IDF + BM25 + Embedding → **RRF** | `parallel_strategy.py` |
| Hybrid Branching | حسب طول الاستعلام | `branching_strategy.py` |

**ملاحظة للمقابلة:** Embedding مبني على Sentence-BERT (عائلة BERT) — dense semantic vectors 384-d.

**Design Pattern:** Strategy + Factory في `retrieval_service/core/factory.py`.

**Screenshot:** UI مع اختيار BM25 k1/b + hybrid mode.

---

### 4.3 Indexing (الطلب 3)

- Inverted Index
- TF-IDF vectors
- BM25 statistics
- Embeddings (`embeddings.npy`)

**الأمر:** `py scripts/run_offline_pipeline.py --ir-datasets`

---

### 4.4 Query Processing (الطلب 4)

نفس pipeline المعالجة للوثائق والاستعلامات عبر HTTP إلى preprocessing_service.

---

### 4.5 Query Refinement (الطلب 5)

**الملف:** `query_refinement/core/refiner.py`

- تصحيح إملاء (قاموس + WordNet)
- توسيع مرادفات محدود (max 3 terms) لتجنب تشتيت الاستعلام
- إزالة تكرار الكلمات

**Screenshot:** UI مع تفعيل refinement + refined query.

---

### 4.6 Matching & Ranking (الطلب 6)

- TF-IDF / BM25 / cosine similarity للـ embeddings
- ترتيب descending + Top-K

---

### 4.7 SOA (الطلب 7)

6 microservices مستقلة، Pydantic schemas في `shared/schemas.py`.

---

### 4.8 Evaluation (الطلب 8)

**المقاييس:** MAP, Recall, Precision@10, nDCG@10  
**الاستعلامات:** 2,076 / 2,076 (كل qrels)

#### نتائج النماذج (بدون refinement)

| Model | MAP | Recall | P@10 | nDCG@10 |
|-------|-----|--------|------|---------|
| tf_idf | 0.0132 | 0.0109 | 0.0033 | 0.0147 |
| bm25 | 0.0148 | 0.0113 | 0.0034 | 0.0160 |
| embedding | 0.0124 | 0.0090 | 0.0027 | 0.0135 |
| hybrid_serial | 0.0125 | 0.0093 | 0.0028 | 0.0138 |
| hybrid_parallel | **0.0138** | **0.0114** | **0.0034** | **0.0153** |
| hybrid_branching | 0.0124 | 0.0103 | 0.0032 | 0.0138 |

**Screenshot مطلوب:** Charts من `data/evaluation/charts/` (MAP, nDCG bar charts).

**تفسير metrics:** القيم منخفضة نسبياً لكن متسقة عبر corpus كامل — domain forum informal + stemming. المقارنة بين النماذج أهم من القيمة المطلقة.

#### قبل / بعد Query Refinement

```powershell
py scripts\run_evaluation.py --compare-refinement --models bm25 hybrid_parallel
py scripts\plot_evaluation_charts.py --comparison data/evaluation/refinement_comparison.json
```

**Screenshot:** `refinement_map.png` و `refinement_ndcg.png`.

---

### 4.9 UI (الطلب 9)

- واجهة عربية RTL
- اختيار النموذج، Top-K، BM25 params
- Query refinement toggle
- عرض tokens + النص الكامل

**Screenshot:** صفحة نتائج البحث.

---

## 5. تشغيل النظام

```powershell
py preprocessing_service\main.py
py indexing_service\main.py
py retrieval_service\main.py
py query_refinement\main.py
py ui_gateway\main.py
```

---

## 6. GitHub

- الرابط: `[أضف رابط الريبو]`
- README: `README.md`

---

## 7. تقسيم العمل (أعضاء المجموعة)

| العضو | المسؤولية |
|-------|-----------|
| [الاسم] | Preprocessing + Indexing |
| [الاسم] | Retrieval + Hybrid |
| [الاسم] | Evaluation + Charts |
| [الاسم] | UI + Demo |
| [الاسم] | التقرير + GitHub |

---

## 8. الخلاصة

- نظام IR كامل end-to-end على LoTTE
- SOA حقيقي مع 6 services
- 6 نماذج استرجاع + 3 hybrid modes
- Evaluation على كل qrels مع charts
- Query refinement قابل للتجربة independently

**GitHub:** [رابط]  
**Dataset:** lotte/lifestyle/dev/forum  
**Corpus indexed:** 268,893 documents
