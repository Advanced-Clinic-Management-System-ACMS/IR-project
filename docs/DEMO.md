# Demo Script — مقابلة 12–15 دقيقة

## قبل العرض (5 دقائق)

```powershell
cd "d:\five year\ir"
py preprocessing_service\main.py    # 8001
py retrieval_service\main.py        # 8003
py query_refinement\main.py         # 8004
py ui_gateway\main.py               # 8000
```

افتح: http://127.0.0.1:8000  
جهّز slides/charts من `data/evaluation/charts/`

---

## الدقيقة 0–2 — Dataset + SOA

1. شغّل `py load_data.py` (أو screenshot جاهز)
2. قل: **268,893 doc · 2,076 qrels · `lotte/lifestyle/dev/forum`**
3. اشرح: 6 services — preprocessing → retrieval → refinement → UI

**جawab Yamen (اسم dataset):**  
> `lotte_lifestyle_dev_forum` = نفس `lotte/lifestyle/dev/forum` — mapping للملفات فقط.

---

## الدقيقة 2–6 — Basic Search

**وضع: أساسي (Basic)**

| # | Model | Query مثال | ملاحظة |
|---|-------|------------|--------|
| 1 | BM25 | `cat health food` | غيّر k1=1.2, b=0.8 |
| 2 | Hybrid Serial | نفس query | BM25 → embedding re-rank |
| 3 | Hybrid Parallel | نفس query | RRF fusion |
| 4 | Hybrid Branching | `why` (قصير) vs `how to train kitten` (طويل) | يختار نموذج مختلف |

---

## الدقيقة 6–9 — Extra Features (مستقل)

**وضع: Extra**

| # | Refinement | Personalization | History | الهدف |
|---|------------|-----------------|---------|--------|
| 1 | ✓ | ✗ | — | Refinement فقط |
| 2 | ✗ | ✓ | `food, vet visit, kitten health` | Personalization فقط |
| 3 | ✓ | ✓ | نفس history | الاثنين معاً |

أظهر: `استعلام محسّن` + `Personalization applied` في النتائج.

---

## الدقيقة 9–13 — Evaluation (الأهم)

1. **Baseline:** `map_comparison.png` + `ndcg_comparison.png`
2. **Before/After Refinement:** `refinement_map.png` + `refinement_ndcg.png`
3. **Before/After Personalization:** `personalization_map.png` + `personalization_ndcg.png`

قل: BM25 MAP ≈ 0.0148 على 2076 queries — corpus forum كامل.

---

## الدقيقة 13–15 — أسئلة متوقعة

| سؤال | جواب |
|------|------|
| Lemmatization? | Porter **Stemming** — اختيار مناسب للـ dataset |
| Hybrid Serial? | `serial_strategy.py` — BM25 candidates ثم re-rank |
| Hybrid Parallel? | 3 scorers + RRF في `parallel_strategy.py` |
| Inverted index? | داخل sklearn/BM25Okapi |
| Basic vs Extra? | Basic = استرجاع فقط؛ Extra = refinement + personalization |

---

## أوامر التقييم (إذا سألوا)

```powershell
py scripts\run_evaluation.py
py scripts\run_evaluation.py --compare-all-extras
py scripts\plot_evaluation_charts.py --comparison data\evaluation\refinement_comparison.json --personalization-comparison data\evaluation\personalization_comparison.json
```
