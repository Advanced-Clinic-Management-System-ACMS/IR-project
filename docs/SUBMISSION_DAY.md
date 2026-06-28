# checklist يوم التسليم والمقابلة (7/3)

## قبل ما تروح (30 دقيقة)

### 1. شغّل الخدمات (4 terminals)

```powershell
cd "d:\five year\ir"
.\.venv\Scripts\Activate.ps1
```

| Terminal | الأمر |
|----------|-------|
| 1 | `py preprocessing_service\main.py` |
| 2 | `py retrieval_service\main.py` *(انتظر Uvicorn 8003)* |
| 3 | `py query_refinement\main.py` |
| 4 | `py ui_gateway\main.py` |

### 2. تحقق من الفهارس

```powershell
Get-Content data\indexes\lotte_lifestyle_dev_forum\metadata.json
```

يجب أن ترى `"document_count": 268893`.

إذا المجلد فارغ:

```powershell
py scripts\run_offline_pipeline.py --ir-datasets
```

*(ساعات — لا تشغّله يوم المقابلة إلا إذا الفهرس مفقود)*

### 3. تحقق من التقييم

```powershell
.\scripts\validate_evaluation.ps1
```

إذا فشل بسبب `report.json MAP ~0.01`:

```powershell
.\scripts\rebuild_baseline_report.ps1
```

> **الوقت المتوقع:** 6 نماذج × 2076 query ≈ **4–8 ساعات** مع `--workers 1`.  
> **شغّله الليلة قبل التسليم** — مو صباح المقابلة.

---

## Demo المقابلة (5–7 دقائق)

### أ) واجهة المستخدم — http://127.0.0.1:8000

1. **Basic mode:** استعلام بسيط → `bm25` → نتائج rank/score/snippet.
2. **غيّر k1/b** بالـ sliders → أعد البحث → اشرح تأثير BM25.
3. **Extra mode:** فعّل Query Refinement → أظهر الاستعلام المحسّن + suggestions.
4. **Personalization:** فعّل + سجل بحث (قطط/حيوانات) → اشرح `personalization_applied`.
5. **Hybrid Parallel + RRF** → قارن مع bm25.

### ب) المعمارية

- افتح `docs/ARCHITECTURE.md` — اشرح SOA + 6 services.
- اذكر Strategy/Factory في retrieval.

### ج) التقييم

- افتح `data/evaluation/EVALUATION_SUMMARY.md`.
- اعرض charts من `data/evaluation/charts/`.
- **السرد الذهبي:** "Personalization نزل MAP لأن history domain (pets) ≠ corpus domain (lifestyle forum) → Query Drift على test collection ثابت."

---

## أسئلة متوقعة + إجابات

| السؤال | الإجابة |
|--------|---------|
| ليش MAP نزل بعد personalization؟ | Query Drift — history عن pets والـ qrels عن lifestyle عام |
| وين Lemmatization؟ | WordNetLemmatizer موجود؛ stem افتراضي للتوافق مع الفهرس (`IR_NLP_MODE=lemma` + re-index) |
| وين Inverted Index؟ | داخلي في sklearn/rank_bm25؛ artifacts على disk |
| ليش workers=1 بالتقييم؟ | 268k docs × parallel = RAM exhaustion (std::bad_alloc) |
| كيف SOA؟ | HTTP بين services؛ Pydantic contracts؛ كل service منفصل |

---

## ملفات التسليم

- [ ] `docs/REPORT_AR.md` — أكمل أسماء المجموعة + GitHub
- [ ] `README.md` — رابط GitHub
- [ ] `data/evaluation/report.json` — bm25 MAP ≥ 0.1
- [ ] `data/evaluation/refinement_comparison.json` — 2076 queries
- [ ] `data/evaluation/personalization_comparison.json` — 2076 queries
- [ ] `data/evaluation/EVALUATION_SUMMARY.md`
- [ ] `data/evaluation/charts/*.png`
- [ ] نسخة تنفيذية جاهزة للتشغيل

---

## أوامر سريعة

```powershell
# إيقاف services معلقة
.\scripts\reset_and_start.ps1

# charts + summary
py scripts\generate_eval_summary.py
py scripts\plot_evaluation_charts.py
py scripts\plot_evaluation_charts.py --comparison data\evaluation\refinement_comparison.json
py scripts\plot_evaluation_charts.py --personalization-comparison data\evaluation\personalization_comparison.json
```

---

## لا تفعل يوم المقابلة

- ❌ `--workers 3` أو أكثر بالتقييم
- ❌ إعادة pipeline كامل بدون حاجة
- ❌ تعديل preprocessing بدون re-index
- ❌ الاعتماد على `report.json` إذا MAP ~0.01
