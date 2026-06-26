# تقرير مشروع نظام استرجاع المعلومات (IR 2026)

**العنوان:** بناء Information Retrieval System  
**Dataset:** `lotte/lifestyle/dev/forum` — 268,893 وثيقة | 2,076 qrels  
**GitHub:** `[أضف رابط الريبو]`  
**المجموعة:** `[5–7 أعضاء — املأ الأسماء في القسم 9]`

---

## 1. مقدمة

تم بناء محرك بحث مخصص وفق متطلبات مقر IR 2026، باستخدام **Python + FastAPI + SOA + Clean Architecture**.  
النظام يسترجع وثائق من **LoTTE Lifestyle Forum** ويدعم 6 نماذج تمثيل + ميزة إضافية **Personalization (#16)**.

> **ملاحظة Dataset:** المعرف الرسمي في ir-datasets هو `lotte/lifestyle/dev/forum`.  
> المجلد المحلي `lotte_lifestyle_dev_forum` = نفس المجموعة بعد استبدال `/` بـ `_` لأسماء الملفات.

---

## 2. Dataset

| البند | القيمة |
|--------|--------|
| المصدر | [ir-datasets](https://ir-datasets.com/lotte) |
| المعرف الرسمي | `lotte/lifestyle/dev/forum` |
| المجلد المحلي | `lotte_lifestyle_dev_forum` |
| عدد الوثائق | 268,893 (> 200K ✓) |
| Qrels | 2,076 queries ✓ |
| التحقق | `py load_data.py` |

**Screenshot:** مخرجات `load_data.py` (doc count + qrels count).

---

## 3. المعمارية (SOA — الطلب 7)

| الخدمة | المنفذ | المسؤولية |
|--------|--------|-----------|
| ui_gateway | 8000 | واجهة تجريبية |
| preprocessing_service | 8001 | معالجة النصوص |
| indexing_service | 8002 | بناء الفهرس |
| retrieval_service | 8003 | البحث والترتيب |
| query_refinement | 8004 | تحسين الاستعلام + Personalization |
| evaluation_service | 8005 | حساب المقاييس |

**Screenshot:** مخطط Mermaid من `docs/ARCHITECTURE.md`.

**Design Patterns:** Strategy + Factory في `retrieval_service/core/`.

---

## 4. تنفيذ الطلبات الأساسية

### 4.1 Pre-Processing (الطلب 1)

**الملف:** `preprocessing_service/core/nlp_engine.py`

| الخطوة | التنفيذ |
|--------|---------|
| Normalization | lowercase، إزالة URLs/أرقام/punctuation |
| Tokenization | NLTK `word_tokenize` |
| Stopwords | NLTK English stopwords |
| Stemming | **Porter Stemmer** (اخترنا Stemming وليس Lemmatization — مناسب لـ forum EN) |

**Screenshot:** كود `stem_tokens()` + tokens في نتائج UI.

---

### 4.2 تمثيل الوثائق (الطلب 2)

| النموذج | التنفيذ | الملف |
|---------|---------|-------|
| TF-IDF | sklearn `TfidfVectorizer` + cosine | `tfidf_strategy.py` |
| BM25 | rank_bm25 `BM25Okapi` | `bm25_strategy.py` |
| Embedding | Sentence-BERT `all-MiniLM-L6-v2` + FAISS | `embedding_strategy.py` |
| Hybrid تسلسلي | BM25 top-100 → embedding re-rank | `serial_strategy.py` |
| Hybrid متوازي | TF-IDF + BM25 + Embedding → RRF / Weighted | `parallel_strategy.py` |
| Hybrid تفرعي | ≤2 tokens→BM25، ≤5→TF-IDF، else→Embedding | `branching_strategy.py` |

**BM25 k1/b:** قابلة للتعديل من UI → تمرير إلى `score_bm25()`.

**Screenshot:** UI — اختيار Hybrid + BM25 params + fusion mode.

---

### 4.3 Indexing (الطلب 3)

- TF-IDF sparse matrix (sklearn)
- BM25 corpus statistics (rank_bm25)
- Sentence-BERT embeddings + FAISS index (offline)
- SQLite `data/documents.db` للنص الأصلي

> الفهرس المعكوس (Inverted Index) مبني **داخلياً** داخل sklearn و rank_bm25 — ليس ملفاً منفصلاً.

**الأمر:** `py scripts/run_offline_pipeline.py --ir-datasets`

---

### 4.4 Query Processing (الطلب 4)

نفس `NLPEngine` للوثائق والاستعلامات عبر HTTP:

`retrieval_service` → `preprocess_client.py` → `/process-query`

---

### 4.5 Query Refinement (الطلب 5)

**الملف:** `query_refinement/core/refiner.py`

- تصحيح إملاء (قاموس + WordNet)
- توسيع مرادفات (max 3 terms)

**تجربة مستقلة:** UI → وضع Extra → ✓ Query Refinement (بدون Personalization).

---

### 4.6 Matching & Ranking (الطلب 6)

- TF-IDF / BM25: dot product / probabilistic scores
- Embedding: cosine via FAISS IndexFlatIP
- Hybrid: serial re-rank أو RRF fusion

---

### 4.7 UI (الطلب 9)

- اختيار Dataset + النموذج + Top-K
- BM25 k1/b + Hybrid fusion
- **وضع Basic:** الطلبات الأساسية فقط
- **وضع Extra:** Refinement و Personalization **بشكل مستقل** (checkbox منفصل لكل ميزة)

**Screenshot:** Basic vs Extra + نتائج البحث.

---

## 5. الميزة الإضافية — Personalization (#16)

**الملف:** `query_refinement/core/personalization.py`

| تقنية IR | الوصف |
|----------|--------|
| History term profile | توسيع الاستعلام بكلمات متكررة من سجل البحث |
| Semantic history matching | cosine similarity مع أقرب استعلام سابق |

**تجربة مستقلة:** UI → Extra → ✓ Personalization + سجل بحث (بدون Refinement).

**Screenshot:** UI مع `personalization_applied` في النتائج.

---

## 6. Evaluation (الطلب 8) — **الأهم**

### 6.1 Baseline — كل النماذج (2,076 queries)

| Model | MAP | Recall | P@10 | nDCG@10 |
|-------|-----|--------|------|---------|
| tf_idf | 0.0132 | 0.0109 | 0.0033 | 0.0147 |
| bm25 | **0.0148** | 0.0113 | 0.0034 | **0.0160** |
| embedding | 0.0124 | 0.0090 | 0.0027 | 0.0135 |
| hybrid_serial | 0.0125 | 0.0093 | 0.0028 | 0.0138 |
| hybrid_parallel | 0.0138 | 0.0114 | 0.0034 | 0.0153 |
| hybrid_branching | 0.0124 | 0.0103 | 0.0032 | 0.0138 |

**Charts:** `data/evaluation/charts/map_comparison.png`, `ndcg_comparison.png`

**Screenshot:** Charts MAP + nDCG لكل النماذج.

**تفسير:** القيم متسقة على corpus كامل (268K forum informal). BM25 الأفضل lexical؛ hybrid_parallel قريب من الأعلى.

---

### 6.2 قبل / بعد Query Refinement

**الأمر:**
```powershell
py scripts\run_evaluation.py --compare-refinement
py scripts\plot_evaluation_charts.py --comparison data\evaluation\refinement_comparison.json
```

**Charts:** `refinement_map.png`, `refinement_ndcg.png`

**Screenshot:** أعمدة before/after لـ MAP و nDCG.

> Refinement قد لا يرفع MAP على qrels لأن ground truth مبني على نص الاستعلام الأصلي — Chart يوضح التأثير objectively.

---

### 6.3 قبل / بعد Personalization (#16)

**الأمر:**
```powershell
py scripts\run_evaluation.py --compare-personalization
py scripts\plot_evaluation_charts.py --personalization-comparison data\evaluation\personalization_comparison.json
```

**Charts:** `personalization_map.png`, `personalization_ndcg.png`

**سجل التقييم:** `DEFAULT_EVAL_HISTORY` في `shared/config.py`

---

### 6.4 تشغيل كل التقييمات دفعة واحدة

```powershell
py scripts\run_evaluation.py
py scripts\run_evaluation.py --compare-all-extras
py scripts\plot_evaluation_charts.py --comparison data\evaluation\refinement_comparison.json --personalization-comparison data\evaluation\personalization_comparison.json
```

---

## 7. تشغيل النظام

```powershell
py preprocessing_service\main.py
py retrieval_service\main.py
py query_refinement\main.py
py ui_gateway\main.py
```

**Demo:** راجع `docs/DEMO.md` (عرض 12–15 دقيقة).

---

## 8. GitHub

- الرابط: `[أضف رابط الريبو]`
- README: `README.md` — بنية الكود + name mapping
- Indexes محلية: `py scripts/migrate_to_library_v2.py` أو pipeline

---

## 9. تقسيم العمل

| العضو | المسؤولية |
|-------|-----------|
| `[الاسم 1]` | preprocessing_service + indexing_service |
| `[الاسم 2]` | retrieval_service + Hybrid strategies |
| `[الاسم 3]` | query_refinement + Personalization |
| `[الاسم 4]` | evaluation + charts |
| `[الاسم 5]` | ui_gateway + demo + تقرير |

> كل عضو يجب أن يشرح **ملفاته** و**3 أسئلة** عنها في المقابلة الفردية.

---

## 10. الخلاصة

- نظام IR end-to-end على LoTTE (268,893 doc, 2076 qrels)
- SOA: 6 services + Clean Architecture
- 6 نماذج + Hybrid Serial / Parallel / Branching
- Stemming (Porter) — مو Lemmatization
- Personalization (#16) — ميزة إضافية قابلة للتجربة مستقلاً
- Evaluation: MAP + nDCG charts + before/after extras

**Dataset ID:** `lotte/lifestyle/dev/forum`  
**Local folder:** `lotte_lifestyle_dev_forum`
