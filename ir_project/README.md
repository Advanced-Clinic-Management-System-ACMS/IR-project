# Information Retrieval System (IR 2026)

Python-based search engine project built with **Service-Oriented Architecture (SOA)** and **FastAPI**.

## Project Structure

```text
ir_project/
├── preprocessing_service/   # NLTK cleaning, tokenization, stemming
├── indexing_service/        # Inverted index, TF-IDF, BM25 stats, embeddings
├── retrieval_service/       # TF-IDF, BM25, Embeddings, Hybrid search
├── query_refinement/        # Query reformulation service
├── evaluation_service/      # MAP, Recall, Precision@10, nDCG
├── ui_gateway/              # Streamlit demo UI
├── shared/                  # Common schemas, config, MongoDB helper
├── scripts/                 # Offline pipeline scripts
└── data/                    # Raw, processed, and index files
```

## Requirements

- Python 3.11+
- MongoDB running locally on `mongodb://localhost:27017`

## Setup

```bash
cd ir_project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Services

Open separate terminals from the `ir_project` folder:

```bash
python -m preprocessing_service.main
python -m indexing_service.main
python -m retrieval_service.main
python -m query_refinement.main
python -m evaluation_service.main
streamlit run ui_gateway/app.py
```

## Build Indexes (Offline)

After services are running:

```bash
python scripts/run_offline_pipeline.py --limit 100
```

For a real dataset from [ir-datasets](https://ir-datasets.com):

```bash
python scripts/run_offline_pipeline.py --ir-datasets --dataset msmarco-passage/train --limit 200000
```

## API Docs

- Preprocessing: http://127.0.0.1:8001/docs
- Indexing: http://127.0.0.1:8002/docs
- Retrieval: http://127.0.0.1:8003/docs
- Query Refinement: http://127.0.0.1:8004/docs
- Evaluation: http://127.0.0.1:8005/docs

## Team Workflow (GitHub)

1. Create one repository for the group.
2. Each member works on a separate service folder or feature branch.
3. Use pull requests instead of pushing directly to `main`.
4. Do not upload full datasets to GitHub; store them locally or with Git LFS if needed.

## Notes

- Raw documents must be read from MongoDB during online search.
- Processed data and indexes can be stored in files during offline processing.
- Hybrid retrieval supports both serial and parallel fusion.
- BM25 parameters `k1` and `b` can be changed from the UI/API.
