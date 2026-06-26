"""
Generate evaluation charts (MAP, nDCG, Recall) from JSON reports.
Safe to run without re-evaluating — reads existing report files only.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_to_dataframe(report: dict) -> pd.DataFrame:
    df = pd.DataFrame(report["metrics"]).T
    df.index.name = "model"
    return df


def plot_model_comparison(
    report: dict,
    output_dir: Path,
    title_prefix: str = "",
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = metrics_to_dataframe(report)
    models = df.index.tolist()

    metric_specs = [
        ("MAP", "Mean Average Precision (MAP)", "map_comparison.png"),
        ("nDCGAt10", "nDCG@10", "ndcg_comparison.png"),
        ("Recall", "Recall", "recall_comparison.png"),
        ("PrecisionAt10", "Precision@10", "precision_comparison.png"),
    ]

    for column, ylabel, filename in metric_specs:
        if column not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        values = df[column].tolist()
        bars = ax.bar(models, values, color="#4C78A8", edgecolor="#2F4B7C")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Retrieval Model")
        ax.set_title(f"{title_prefix}{ylabel} — LoTTE qrels ({report.get('evaluated_queries', '?')} queries)")
        ax.tick_params(axis="x", rotation=35)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)

    combined = df[["MAP", "nDCGAt10", "Recall"]].copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(models))
    width = 0.25
    ax.bar([i - width for i in x], combined["MAP"], width=width, label="MAP")
    ax.bar(x, combined["nDCGAt10"], width=width, label="nDCG@10")
    ax.bar([i + width for i in x], combined["Recall"], width=width, label="Recall")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"{title_prefix}Model Comparison — MAP / nDCG@10 / Recall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "combined_metrics.png", dpi=150)
    plt.close(fig)

    return df


def plot_refinement_comparison(
    comparison: dict,
    output_dir: Path,
    title_prefix: str = "",
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for model_name, pair in comparison["models"].items():
        rows.append({"model": model_name, "phase": "before", **pair["before"]})
        rows.append({"model": model_name, "phase": "after", **pair["after"]})
    df = pd.DataFrame(rows)

    comparison_type = comparison.get("comparison_type", "before_after")
    label = title_prefix or comparison_type.replace("_", " ").title()
    prefix = "personalization" if "personalization" in comparison_type else "refinement"

    for metric, filename_suffix, ylabel in [
        ("MAP", f"{prefix}_map.png", "MAP"),
        ("nDCGAt10", f"{prefix}_ndcg.png", "nDCG@10"),
    ]:
        pivot = df.pivot(index="model", columns="phase", values=metric)
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot.plot(kind="bar", ax=ax, color=["#E45756", "#54A24B"])
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Model")
        ax.set_title(f"{label} — {ylabel}")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Phase")
        fig.tight_layout()
        fig.savefig(output_dir / filename_suffix, dpi=150)
        plt.close(fig)

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot evaluation charts from JSON reports.")
    parser.add_argument("--report", default="data/evaluation/report.json")
    parser.add_argument("--comparison", default=None, help="refinement_comparison.json path")
    parser.add_argument(
        "--personalization-comparison",
        default=None,
        help="personalization_comparison.json path",
    )
    parser.add_argument("--output", default="data/evaluation/charts")
    args = parser.parse_args()

    report_path = Path(args.report)
    if report_path.exists():
        report = load_report(report_path)
        df = plot_model_comparison(report, Path(args.output))
        print(df)
        print(f"Saved charts to {args.output}/")

    for comp_arg in (args.comparison, args.personalization_comparison):
        if comp_arg:
            comp_path = Path(comp_arg)
            if comp_path.exists():
                plot_refinement_comparison(load_report(comp_path), Path(args.output))
                print(f"Saved comparison charts from {comp_path} to {args.output}/")
