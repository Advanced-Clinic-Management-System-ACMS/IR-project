"""
Generate interview-ready evaluation summary tables from JSON reports.
Reads existing files only — no re-evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_baseline_table(report: dict) -> str:
    lines = [
        "# Baseline Evaluation (All Models)",
        "",
        f"- Dataset: `{report.get('dataset', '?')}`",
        f"- Total qrels queries: **{report.get('total_qrels_queries', '?')}**",
        f"- Evaluated queries: **{report.get('evaluated_queries', '?')}**",
        f"- Top-K: {report.get('top_k', 10)}",
        "",
        "| Model | MAP | Recall | P@10 | nDCG@10 |",
        "|-------|-----|--------|------|---------|",
    ]
    for model, metrics in report.get("metrics", {}).items():
        lines.append(
            f"| {model} | {metrics['MAP']:.4f} | {metrics['Recall']:.4f} | "
            f"{metrics['PrecisionAt10']:.4f} | {metrics['nDCGAt10']:.4f} |"
        )
    return "\n".join(lines)


def format_comparison_table(comparison: dict, title: str) -> str:
    comp_type = comparison.get("comparison_type", "before_after")
    lines = [
        f"# {title}",
        "",
        f"- Dataset: `{comparison.get('dataset', '?')}`",
        f"- Total qrels queries: **{comparison.get('total_qrels_queries', '?')}**",
        f"- Evaluated queries: **{comparison.get('evaluated_queries', '?')}**",
        f"- Comparison: `{comp_type}`",
        "",
        "| Model | Phase | MAP | Recall | P@10 | nDCG@10 | Δ MAP | Δ nDCG@10 |",
        "|-------|-------|-----|--------|------|---------|-------|-----------|",
    ]
    for model, pair in comparison.get("models", {}).items():
        before = pair["before"]
        after = pair["after"]
        delta_map = pair.get("delta_MAP", after["MAP"] - before["MAP"])
        delta_ndcg = pair.get("delta_nDCGAt10", after["nDCGAt10"] - before["nDCGAt10"])
        lines.append(
            f"| {model} | before | {before['MAP']:.4f} | {before['Recall']:.4f} | "
            f"{before['PrecisionAt10']:.4f} | {before['nDCGAt10']:.4f} | — | — |"
        )
        lines.append(
            f"| {model} | after | {after['MAP']:.4f} | {after['Recall']:.4f} | "
            f"{after['PrecisionAt10']:.4f} | {after['nDCGAt10']:.4f} | "
            f"{delta_map:+.4f} | {delta_ndcg:+.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    eval_dir = Path("data/evaluation")
    output_md = eval_dir / "EVALUATION_SUMMARY.md"
    sections: list[str] = [
        "# IR Evaluation Summary — Ready for Interview",
        "",
        "All metrics computed offline on the **full qrels file** (no sampling).",
        "",
    ]

    report_path = eval_dir / "report.json"
    if report_path.exists():
        sections.append(format_baseline_table(load_json(report_path)))
        sections.append("")

    refinement_path = eval_dir / "refinement_comparison.json"
    if refinement_path.exists():
        sections.append(
            format_comparison_table(
                load_json(refinement_path),
                "Query Refinement — Before vs After",
            )
        )
        sections.append("")

    personalization_path = eval_dir / "personalization_comparison.json"
    if personalization_path.exists():
        sections.append(
            format_comparison_table(
                load_json(personalization_path),
                "Personalization (#16) — Before vs After",
            )
        )
        sections.append("")

    sections.append("## Charts")
    sections.append("")
    sections.append("- `data/evaluation/charts/map_comparison.png`")
    sections.append("- `data/evaluation/charts/ndcg_comparison.png`")
    sections.append("- `data/evaluation/charts/refinement_map.png`")
    sections.append("- `data/evaluation/charts/personalization_map.png` (if generated)")

    output_md.write_text("\n".join(sections), encoding="utf-8")
    print(f"Saved {output_md}")


if __name__ == "__main__":
    main()
