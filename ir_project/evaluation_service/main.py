import math

from fastapi import FastAPI

from shared.schemas import HealthResponse

app = FastAPI(
    title="Evaluation Service",
    description="Compute MAP, Recall, Precision@10, and nDCG.",
    version="1.0.0",
)


def dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def ndcg_at_k(relevances: list[int], k: int) -> float:
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg(relevances) / ideal_dcg


def average_precision(relevances: list[int]) -> float:
    hits = 0
    precision_sum = 0.0
    for idx, rel in enumerate(relevances, start=1):
        if rel:
            hits += 1
            precision_sum += hits / idx
    if hits == 0:
        return 0.0
    return precision_sum / hits


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(service="evaluation_service", status="ok")


@app.post("/evaluate")
def evaluate(payload: dict) -> dict:
    """
    Expected payload:
    {
      "runs": {
        "tf_idf": {"q1": ["d1", "d2"], "q2": ["d3"]},
        "bm25": {...}
      },
      "qrels": {
        "q1": {"d1": 1, "d4": 1},
        "q2": {"d3": 1}
      },
      "k": 10
    }
    """
    runs: dict[str, dict[str, list[str]]] = payload.get("runs", {})
    qrels: dict[str, dict[str, int]] = payload.get("qrels", {})
    k = int(payload.get("k", 10))

    report: dict[str, dict[str, float]] = {}

    for model_name, run in runs.items():
        ap_values: list[float] = []
        recall_values: list[float] = []
        precision_values: list[float] = []
        ndcg_values: list[float] = []

        for query_id, ranked_docs in run.items():
            relevant = qrels.get(query_id, {})
            rel_set = {doc_id for doc_id, rel in relevant.items() if rel > 0}
            if not rel_set:
                continue

            ranked = ranked_docs[:k]
            binary_relevances = [1 if doc_id in rel_set else 0 for doc_id in ranked]

            ap_values.append(average_precision(binary_relevances))
            recall_values.append(len(set(ranked) & rel_set) / len(rel_set))
            precision_values.append(sum(binary_relevances) / k)
            ndcg_values.append(
                ndcg_at_k(
                    [relevant.get(doc_id, 0) for doc_id in ranked],
                    k,
                )
            )

        report[model_name] = {
            "MAP": round(sum(ap_values) / len(ap_values), 4) if ap_values else 0.0,
            "Recall": round(sum(recall_values) / len(recall_values), 4) if recall_values else 0.0,
            "Precision@10": round(sum(precision_values) / len(precision_values), 4)
            if precision_values
            else 0.0,
            "nDCG@10": round(sum(ndcg_values) / len(ndcg_values), 4) if ndcg_values else 0.0,
        }

    return {"k": k, "metrics": report}


if __name__ == "__main__":
    import uvicorn

    from shared.config import SERVICE_PORTS

    uvicorn.run(
        "evaluation_service.main:app",
        host="127.0.0.1",
        port=SERVICE_PORTS["evaluation"],
        reload=True,
    )
