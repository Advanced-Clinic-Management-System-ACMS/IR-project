# Information Retrieval System (IR 2026) — LoTTE + SOA

Python search engine with **Service-Oriented Architecture (SOA)** and **Clean Architecture**.

**Dataset:** `lotte/lifestyle/dev/forum` from [ir-datasets](https://ir-datasets.com/lotte)  
**Download location:** `C:\Users\<USER>\.ir_datasets\lotte` (~4GB)

## Storage model (file-based, no MongoDB)

| What | Where |
|------|-------|
| LoTTE source files | `~/.ir_datasets/lotte` |
| Raw document text | `data/raw/lotte_lifestyle_dev_forum.jsonl` |
| Processed tokens | `data/processed/lotte_lifestyle_dev_forum.jsonl` |
| Indexes (TF-IDF, BM25, Sentence-BERT embeddings) | `data/indexes/lotte_lifestyle_dev_forum/` |

At **query time**, retrieval reads **compressed indexes from disk** and **original document text from SQLite** (`data/documents.db`).

## Services

| Service | Port | Role |
|---------|------|------|
| preprocessing_service | 8001 | NLTK normalization, tokenization, stemming |
| indexing_service | 8002 | Inverted index, TF-IDF, BM25 stats, Sentence-BERT embeddings |
| retrieval_service | 8003 | Strategy + Factory pattern; TF-IDF, BM25, Embedding, Hybrid |
| query_refinement | 8004 | Spelling correction + WordNet synonym expansion |
| evaluation_service | 8005 | MAP, Recall, P@10, nDCG |
| ui_gateway | 8000 | Web UI (FastAPI + Jinja2) |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the system diagram.

## Setup

```powershell
cd "d:\five year\ir"
py -m pip install -r requirements.txt
py load_data.py
```

## Run (correct order)

### 1) Start backend services (4 terminals)

```powershell
cd "d:\five year\ir"
py preprocessing_service\main.py
py indexing_service\main.py
py retrieval_service\main.py
py query_refinement\main.py
```

### 2) Build indexes — full LoTTE corpus

```powershell
py scripts\run_offline_pipeline.py --ir-datasets
```

Quick test (500 docs):

```powershell
py scripts\run_offline_pipeline.py --ir-datasets --limit 500
```

Resume after interruption:

```powershell
py scripts\run_offline_pipeline.py --ir-datasets --resume
```

### 3) Start UI

```powershell
py ui_gateway\main.py
```

Open: http://127.0.0.1:8000

## Retrieval models

| Model | Implementation |
|-------|----------------|
| TF-IDF | VSM cosine similarity |
| BM25 | Probabilistic ranking (`k1`, `b` from UI) |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` (384-d dense vectors) |
| Hybrid Serial | BM25 top-100 → embedding re-ranking |
| Hybrid Parallel | TF-IDF + BM25 + Embedding in parallel → **RRF fusion** |
| Hybrid Branching | Model chosen by query length |

Production path: `retrieval_service/core/factory.py` → `core/strategies/` → `core/scoring.py`

## Evaluation (all 2076 qrels queries)

```powershell
# Full pipeline (multithreaded retrieval, interview-ready outputs)
powershell -ExecutionPolicy Bypass -File scripts\run_full_report_eval.ps1

# Validate before submission:
powershell -ExecutionPolicy Bypass -File scripts\validate_evaluation.ps1

# Rebuild stale baseline only (if MAP ~0.01):
powershell -ExecutionPolicy Bypass -File scripts\rebuild_baseline_report.ps1

# Or manual:
py scripts\run_evaluation.py --workers 1
py scripts\run_evaluation.py --compare-all-extras --workers 1
py scripts\plot_evaluation_charts.py --comparison data\evaluation\refinement_comparison.json --personalization-comparison data\evaluation\personalization_comparison.json
py scripts\generate_eval_summary.py
```

Interview-ready outputs: `data/evaluation/EVALUATION_SUMMARY.md` + `data/evaluation/charts/*.png`  
Submission checklist: `docs/SUBMISSION_DAY.md` | Arabic report: `docs/REPORT_AR.md`

Use `--query-limit 100` only for quick local testing — never in the report.

## Name mapping

```
ir-datasets ID : lotte/lifestyle/dev/forum
local folder   : lotte_lifestyle_dev_forum
```
