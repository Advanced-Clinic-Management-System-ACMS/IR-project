"""
Run IR evaluation against LoTTE qrels using the retrieval service.
Prints query counts and metrics for each retrieval model.

Uses multithreaded retrieval (HTTP I/O) to speed up full 2076-query runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if sys.platform == "win32" and not sys.flags.utf8_mode:
    import subprocess

    raise SystemExit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ir_datasets
import requests

from evaluation_service.schemas.payloads import EvaluateRequest
from evaluation_service.services.evaluator import EvaluationService
from shared.config import (
    DEFAULT_DATASET_NAME,
    DEFAULT_EVAL_HISTORY,
    DEFAULT_IR_DATASET,
    SERVICE_URLS,
)
from shared.schemas import RetrievalModel, SearchRequest

DEFAULT_EVAL_WORKERS = 1
SERVICE_HEALTH_TIMEOUT = 120


def check_service(name: str, url: str, timeout: int = SERVICE_HEALTH_TIMEOUT) -> None:
    try:
        response = requests.get(f"{url}/health", timeout=timeout)
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
    use_personalization: bool = False,
    user_history: list[str] | None = None,
    max_retries: int = 3,
) -> list[str]:
    payload = SearchRequest(
        query=query_text,
        model=model,
        top_k=top_k,
        dataset_name=DEFAULT_DATASET_NAME,
        use_refinement=use_refinement,
        use_personalization=use_personalization,
        user_history=user_history or [],
    )
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                f"{SERVICE_URLS['retrieval']}/search",
                json=payload.model_dump(mode="json"),
                timeout=120,
            )
            response.raise_for_status()
            return [item["doc_id"] for item in response.json()["results"]]
        except requests.RequestException as exc:
            last_error = exc
            detail = ""
            if hasattr(exc, "response") and exc.response is not None:
                detail = exc.response.text[:300]
            print(
                f"  [retry {attempt}/{max_retries}] retrieval failed: {exc} {detail}",
                flush=True,
            )
            if attempt < max_retries:
                import time

                time.sleep(5 * attempt)
    raise last_error  # type: ignore[misc]


def _retrieve_task(
    query_id: str,
    query_text: str,
    model: RetrievalModel,
    top_k: int,
    retrieve_kwargs: dict,
) -> tuple[str, list[str]]:
    doc_ids = retrieve(query_text, model, top_k, **retrieve_kwargs)
    return query_id, doc_ids


def build_run_parallel(
    query_ids: list[str],
    queries: dict[str, str],
    model: RetrievalModel,
    top_k: int,
    workers: int,
    retrieve_kwargs: dict | None = None,
    progress_label: str = "",
) -> dict[str, list[str]]:
    retrieve_kwargs = retrieve_kwargs or {}
    run: dict[str, list[str]] = {}
    total = len(query_ids)
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _retrieve_task,
                query_id,
                queries[query_id],
                model,
                top_k,
                retrieve_kwargs,
            ): query_id
            for query_id in query_ids
        }
        for future in as_completed(futures):
            query_id, doc_ids = future.result()
            run[query_id] = doc_ids
            completed += 1
            if completed == 1 or completed % 50 == 0 or completed == total:
                prefix = f"{progress_label} " if progress_label else ""
                print(f"  {prefix}Query {completed}/{total}")

    return run


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
    workers: int = DEFAULT_EVAL_WORKERS,
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
    print(f"Parallel workers: {workers}")
    print(f"Query refinement: {'ON' if use_refinement else 'OFF'}")
    print("=" * 70)

    all_metrics: dict[str, dict[str, float]] = {}
    for model in models:
        print(f"\nRunning model: {model.value}")
        run = build_run_parallel(
            query_ids=query_ids,
            queries=queries,
            model=model,
            top_k=top_k,
            workers=workers,
            retrieve_kwargs={"use_refinement": use_refinement},
        )

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
        "workers": workers,
        "use_refinement": use_refinement,
        "metrics": all_metrics,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved report to {output_path}")
    return payload


def run_before_after_comparison(
    dataset_name: str,
    models: list[RetrievalModel],
    top_k: int,
    query_limit: int | None,
    output_path: Path | None,
    use_api: bool,
    comparison_type: str,
    after_kwargs: dict,
    workers: int = DEFAULT_EVAL_WORKERS,
) -> dict:
    check_service("retrieval", SERVICE_URLS["retrieval"])
    if after_kwargs.get("use_refinement") or after_kwargs.get("use_personalization"):
        check_service("query_refinement", SERVICE_URLS["query_refinement"])
    evaluate_fn = evaluate_runs_api if use_api else evaluate_runs_local

    qrels = load_qrels(dataset_name)
    queries = load_queries(dataset_name, qrels)
    query_ids = sorted(queries.keys())
    if query_limit:
        query_ids = query_ids[:query_limit]

    print("=" * 70)
    print(f"{comparison_type.upper()} COMPARISON (before vs after)")
    print(f"Dataset: {dataset_name}")
    print(f"Queries: {len(query_ids)}")
    print(f"Parallel workers: {workers}")
    print(f"Models: {[m.value for m in models]}")
    print(f"After settings: {after_kwargs}")
    print("=" * 70)

    comparison: dict[str, dict] = {"models": {}}
    for model in models:
        print(f"\nModel: {model.value} — BEFORE (baseline)")
        run_before = build_run_parallel(
            query_ids=query_ids,
            queries=queries,
            model=model,
            top_k=top_k,
            workers=workers,
            progress_label="before",
        )
        print(f"Model: {model.value} — AFTER ({comparison_type})")
        run_after = build_run_parallel(
            query_ids=query_ids,
            queries=queries,
            model=model,
            top_k=top_k,
            workers=workers,
            retrieve_kwargs=after_kwargs,
            progress_label="after",
        )

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
        "workers": workers,
        "comparison_type": comparison_type,
        "after_settings": after_kwargs,
        "models": comparison["models"],
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved comparison to {output_path}")
    return payload


def run_refinement_comparison(
    dataset_name: str,
    models: list[RetrievalModel],
    top_k: int,
    query_limit: int | None,
    output_path: Path | None,
    use_api: bool,
    workers: int = DEFAULT_EVAL_WORKERS,
) -> dict:
    return run_before_after_comparison(
        dataset_name=dataset_name,
        models=models,
        top_k=top_k,
        query_limit=query_limit,
        output_path=output_path,
        use_api=use_api,
        comparison_type="query_refinement_before_after",
        after_kwargs={"use_refinement": True},
        workers=workers,
    )


def run_personalization_comparison(
    dataset_name: str,
    models: list[RetrievalModel],
    top_k: int,
    query_limit: int | None,
    output_path: Path | None,
    use_api: bool,
    history: list[str],
    workers: int = DEFAULT_EVAL_WORKERS,
) -> dict:
    return run_before_after_comparison(
        dataset_name=dataset_name,
        models=models,
        top_k=top_k,
        query_limit=query_limit,
        output_path=output_path,
        use_api=use_api,
        comparison_type="personalization_before_after",
        after_kwargs={
            "use_personalization": True,
            "user_history": history,
        },
        workers=workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval models on LoTTE qrels.")
    parser.add_argument("--dataset", default=DEFAULT_IR_DATASET)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--query-limit",
        type=int,
        default=None,
        help="Use all qrels queries if omitted (2076 for LoTTE lifestyle forum)",
    )
    parser.add_argument("--output", default="data/evaluation/report.json")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_EVAL_WORKERS,
        help="Parallel threads for retrieval HTTP calls",
    )
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
        help="Run before/after query refinement comparison",
    )
    parser.add_argument(
        "--compare-personalization",
        action="store_true",
        help="Run before/after personalization comparison (Requirement #16)",
    )
    parser.add_argument(
        "--comparison-output",
        default="data/evaluation/refinement_comparison.json",
    )
    parser.add_argument(
        "--personalization-output",
        default="data/evaluation/personalization_comparison.json",
    )
    parser.add_argument(
        "--compare-all-extras",
        action="store_true",
        help="Run both refinement and personalization before/after comparisons",
    )
    args = parser.parse_args()
    models = [RetrievalModel(name) for name in args.models]

    comparison_models = (
        [RetrievalModel.BM25, RetrievalModel.HYBRID_PARALLEL]
        if set(args.models) == {m.value for m in RetrievalModel}
        else models
    )

    if args.compare_all_extras:
        run_refinement_comparison(
            dataset_name=args.dataset,
            models=comparison_models,
            top_k=args.top_k,
            query_limit=args.query_limit,
            output_path=Path(args.comparison_output),
            use_api=args.use_api,
            workers=args.workers,
        )
        run_personalization_comparison(
            dataset_name=args.dataset,
            models=comparison_models,
            top_k=args.top_k,
            query_limit=args.query_limit,
            output_path=Path(args.personalization_output),
            use_api=args.use_api,
            history=DEFAULT_EVAL_HISTORY,
            workers=args.workers,
        )
    elif args.compare_refinement:
        run_refinement_comparison(
            dataset_name=args.dataset,
            models=comparison_models,
            top_k=args.top_k,
            query_limit=args.query_limit,
            output_path=Path(args.comparison_output),
            use_api=args.use_api,
            workers=args.workers,
        )
    elif args.compare_personalization:
        run_personalization_comparison(
            dataset_name=args.dataset,
            models=comparison_models,
            top_k=args.top_k,
            query_limit=args.query_limit,
            output_path=Path(args.personalization_output),
            use_api=args.use_api,
            history=DEFAULT_EVAL_HISTORY,
            workers=args.workers,
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
            workers=args.workers,
        )
