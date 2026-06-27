# IR Evaluation Summary — Ready for Interview

All metrics computed offline on the **full qrels file** (no sampling).

# Baseline Evaluation (All Models)

- Dataset: `lotte/lifestyle/dev/forum`
- Total qrels queries: **2076**
- Evaluated queries: **2076**
- Top-K: 10

| Model | MAP | Recall | P@10 | nDCG@10 |
|-------|-----|--------|------|---------|
| tf_idf | 0.0132 | 0.0109 | 0.0033 | 0.0147 |
| bm25 | 0.0148 | 0.0113 | 0.0034 | 0.0160 |
| embedding | 0.0124 | 0.0090 | 0.0027 | 0.0135 |
| hybrid_serial | 0.0125 | 0.0093 | 0.0028 | 0.0138 |
| hybrid_parallel | 0.0138 | 0.0114 | 0.0034 | 0.0153 |
| hybrid_branching | 0.0124 | 0.0103 | 0.0032 | 0.0138 |

# Query Refinement — Before vs After

- Dataset: `lotte/lifestyle/dev/forum`
- Total qrels queries: **2076**
- Evaluated queries: **2076**
- Comparison: `query_refinement_before_after`

| Model | Phase | MAP | Recall | P@10 | nDCG@10 | Δ MAP | Δ nDCG@10 |
|-------|-------|-----|--------|------|---------|-------|-----------|
| bm25 | before | 0.3176 | 0.2183 | 0.1174 | 0.3923 | — | — |
| bm25 | after | 0.2824 | 0.1928 | 0.1034 | 0.3517 | -0.0352 | -0.0406 |
| hybrid_parallel | before | 0.4055 | 0.2874 | 0.1579 | 0.4915 | — | — |
| hybrid_parallel | after | 0.3723 | 0.2639 | 0.1430 | 0.4545 | -0.0332 | -0.0370 |

# Personalization (#16) — Before vs After

- Dataset: `lotte/lifestyle/dev/forum`
- Total qrels queries: **2076**
- Evaluated queries: **2076**
- Comparison: `personalization_before_after`

| Model | Phase | MAP | Recall | P@10 | nDCG@10 | Δ MAP | Δ nDCG@10 |
|-------|-------|-----|--------|------|---------|-------|-----------|
| bm25 | before | 0.3176 | 0.2183 | 0.1174 | 0.3923 | — | — |
| bm25 | after | 0.2189 | 0.1383 | 0.0706 | 0.2696 | -0.0987 | -0.1227 |
| hybrid_parallel | before | 0.4055 | 0.2874 | 0.1579 | 0.4915 | — | — |
| hybrid_parallel | after | 0.3055 | 0.2059 | 0.1082 | 0.3705 | -0.1000 | -0.1210 |

## Charts

- `data/evaluation/charts/map_comparison.png`
- `data/evaluation/charts/ndcg_comparison.png`
- `data/evaluation/charts/refinement_map.png`
- `data/evaluation/charts/personalization_map.png` (if generated)