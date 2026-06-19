"""
Run IR evaluation against LoTTE qrels using the retrieval service.
Prints query counts and metrics for each retrieval model.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32" and not sys.flags.utf8_mode:
    import os

    os.execv(sys.executable, [sys.executable, "-X", "utf8", *sys.argv])

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ir_datasets
import requests

from evaluation_service.schemas.payloads import EvaluateRequest
from evaluation_service.services.evaluator import EvaluationService
from shared.config import DEFAULT_DATASET_NAME, DEFAULT_IR_DATASET, SERVICE_URLS
from shared.schemas import RetrievalModel, SearchRequest


def check_service(name: str, url: str) -> None:
    try:
        response = requests.get(f"{url}/health", timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(
            f"Service '{name}' is not running at {url}. "
            f"Start it first, then rerun evaluation. Error: {exc}"
        ) from exc


def load_qrels(dataset_name: str) -> dict[str, dict[str, int]]:
    dataset = ir_datasets.load(dataset_name)
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for qrel in dataset.qrels_iter():
        qrels[str(qrel.query_id)][str(qrel.doc_id)] = int(qrel.relevance)
    return dict(qrels)


def load_queries(dataset_name: str, qrels: dict[str, dict[str, int]]) -> dict[str, str]:
    dataset = ir_datasets.load(dataset_name)
    needed = set(qrels.keys())
    queries: dict[str, str] = {}
    for query in dataset.queries_iter():
        query_id = str(query.query_id)
        if query_id in needed:
            queries[query_id] = query.text
        if len(queries) == len(needed):
            break
    return queries


def retrieve(
    query_text: str,
    model: RetrievalModel,
    top_k: int,
    use_refinement: bool = False,
) -> list[str]:
    payload = SearchRequest(
        query=query_text,
        model=model,
        top_k=top_k,
        dataset_name=DEFAULT_DATASET_NAME,
        use_refinement=use_refinement,
    )
    response = requests.post(
        f"{SERVICE_URLS['retrieval']}/search",
        json=payload.model_dump(mode="json"),
        timeout=120,
    )
    response.raise_for_status()
    return [item["doc_id"] for item in response.json()["results"]]


def evaluate_runs_local(
    runs: dict[str, dict[str, list[str]]],
    qrels: dict[str, dict[str, int]],
    k: int,
) -> dict[str, dict[str, float]]:
    report = EvaluationService.generate_report(
        EvaluateRequest(runs=runs, qrels=qrels, k=k)
    )
    return {name: metrics.model_dump() for name, metrics in report.items()}


def evaluate_runs_api(
    runs: dict[str, dict[str, list[str]]],
    qrels: dict[str, dict[str, int]],
    k: int,
) -> dict[str, dict[str, float]]:
    response = requests.post(
        f"{SERVICE_URLS['evaluation']}/evaluate",
        json={"runs": runs, "qrels": qrels, "k": k},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["metrics"]


def run_evaluation(
    dataset_name: str,
    models: list[RetrievalModel],
    top_k: int,
    query_limit: int | None,
    output_path: Path | None,
    use_api: bool,
    use_refinement: bool = False,
) -> dict:
    check_service("retrieval", SERVICE_URLS["retrieval"])
    if use_api:
        check_service("evaluation", SERVICE_URLS["evaluation"])
        evaluate_fn = evaluate_runs_api
        print("Metrics backend: evaluation_service API (port 8005)")
    else:
        evaluate_fn = evaluate_runs_local
        print("Metrics backend: local EvaluationService (no port 8005 required)")

    qrels = load_qrels(dataset_name)
    queries = load_queries(dataset_name, qrels)
    query_ids = sorted(queries.keys())
    if query_limit:
        query_ids = query_ids[:query_limit]

    print("=" * 70)
    print(f"Dataset: {dataset_name}")
    print(f"Total queries in qrels file: {len(qrels)}")
    print(f"Queries used in this evaluation run: {len(query_ids)}")
    print(f"Top-K: {top_k}")
    print(f"Query refinement: {'ON' if use_refinement else 'OFF'}")
    print("=" * 70)

    all_metrics: dict[str, dict[str, float]] = {}
    for model in models:
        print(f"\nRunning model: {model.value}")
        run: dict[str, list[str]] = {}
        for idx, query_id in enumerate(query_ids, start=1):
            if idx % 25 == 0 or idx == 1:
                print(f"  Query {idx}/{len(query_ids)}")
            run[query_id] = retrieve(queries[query_id], model, top_k, use_refinement)

        metrics = evaluate_fn({model.value: run}, qrels, top_k)[model.value]
        all_metrics[model.value] = metrics
        print(
            f"  MAP={metrics['MAP']} Recall={metrics['Recall']} "
            f"P@{top_k}={metrics['PrecisionAt10']} nDCG={metrics['nDCGAt10']}"
        )

    print("\n" + "=" * 70)
    print("FINAL METRICS")
    for model_name, metrics in all_metrics.items():
        print(
            f"{model_name:20} MAP={metrics['MAP']:.4f} Recall={metrics['Recall']:.4f} "
            f"P@10={metrics['PrecisionAt10']:.4f} nDCG@10={metrics['nDCGAt10']:.4f}"
        )
    print("=" * 70)

    payload = {
        "dataset": dataset_name,
        "total_qrels_queries": len(qrels),
        "evaluated_queries": len(query_ids),
        "top_k": top_k,
        "use_refinement": use_refinement,
        "metrics": all_metrics,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved report to {output_path}")
    return payload


def run_refinement_comparison(
    dataset_name: str,
    models: list[RetrievalModel],
    top_k: int,
    query_limit: int | None,
    output_path: Path | None,
    use_api: bool,
) -> dict:
    """Compare retrieval metrics before vs after query refinement (same queries)."""
    check_service("retrieval", SERVICE_URLS["retrieval"])
    check_service("query_refinement", SERVICE_URLS["query_refinement"])
    evaluate_fn = evaluate_runs_api if use_api else evaluate_runs_local

    qrels = load_qrels(dataset_name)
    queries = load_queries(dataset_name, qrels)
    query_ids = sorted(queries.keys())
    if query_limit:
        query_ids = query_ids[:query_limit]

    print("=" * 70)
    print("REFINEMENT COMPARISON (before vs after)")
    print(f"Dataset: {dataset_name}")
    print(f"Queries: {len(query_ids)}")
    print(f"Models: {[m.value for m in models]}")
    print("=" * 70)

    comparison: dict[str, dict] = {"models": {}}
    for model in models:
        print(f"\nModel: {model.value}")
        run_before: dict[str, list[str]] = {}
        run_after: dict[str, list[str]] = {}
        for idx, query_id in enumerate(query_ids, start=1):
            if idx % 25 == 0 or idx == 1:
                print(f"  Query {idx}/{len(query_ids)}")
            text = queries[query_id]
            run_before[query_id] = retrieve(text, model, top_k, use_refinement=False)
            run_after[query_id] = retrieve(text, model, top_k, use_refinement=True)

        before_metrics = evaluate_fn({model.value: run_before}, qrels, top_k)[model.value]
        after_metrics = evaluate_fn({model.value: run_after}, qrels, top_k)[model.value]
        comparison["models"][model.value] = {
            "before": before_metrics,
            "after": after_metrics,
            "delta_MAP": round(after_metrics["MAP"] - before_metrics["MAP"], 6),
            "delta_nDCGAt10": round(after_metrics["nDCGAt10"] - before_metrics["nDCGAt10"], 6),
        }
        print(
            f"  MAP:  {before_metrics['MAP']:.4f} -> {after_metrics['MAP']:.4f} "
            f"(delta {comparison['models'][model.value]['delta_MAP']:+.4f})"
        )
        print(
            f"  nDCG: {before_metrics['nDCGAt10']:.4f} -> {after_metrics['nDCGAt10']:.4f} "
            f"(delta {comparison['models'][model.value]['delta_nDCGAt10']:+.4f})"
        )

    payload = {
        "dataset": dataset_name,
        "total_qrels_queries": len(qrels),
        "evaluated_queries": len(query_ids),
        "top_k": top_k,
        "comparison_type": "query_refinement_before_after",
        "models": comparison["models"],
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved comparison to {output_path}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval models on LoTTE qrels.")
    parser.add_argument("--dataset", default=DEFAULT_IR_DATASET)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--query-limit", type=int, default=None, help="Use all qrels queries if omitted")
    parser.add_argument("--output", default="data/evaluation/report.json")
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Send metrics to evaluation_service on port 8005 instead of computing locally",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[m.value for m in RetrievalModel],
        choices=[m.value for m in RetrievalModel],
    )
    parser.add_argument(
        "--use-refinement",
        action="store_true",
        help="Enable query refinement during retrieval",
    )
    parser.add_argument(
        "--compare-refinement",
        action="store_true",
        help="Run before/after query refinement comparison (uses --models subset)",
    )
    parser.add_argument(
        "--comparison-output",
        default="data/evaluation/refinement_comparison.json",
    )
    args = parser.parse_args()
    models = [RetrievalModel(name) for name in args.models]

    if args.compare_refinement:
        if set(args.models) == {m.value for m in RetrievalModel}:
            comparison_models = [RetrievalModel.BM25, RetrievalModel.HYBRID_PARALLEL]
        else:
            comparison_models = models
        run_refinement_comparison(
            dataset_name=args.dataset,
            models=comparison_models,
            top_k=args.top_k,
            query_limit=args.query_limit,
            output_path=Path(args.comparison_output),
            use_api=args.use_api,
        )
    else:
        run_evaluation(
            dataset_name=args.dataset,
            models=models,
            top_k=args.top_k,
            query_limit=args.query_limit,
            output_path=Path(args.output),
            use_api=args.use_api,
            use_refinement=args.use_refinement,
        )
