"""
Error analysis across experiment results — categorises errors into
dissertation-required types and generates a summary suitable for the
Results chapter.

Categories (per project specification):
    - missed_risk          (FN) ground-truth practice not predicted
    - irrelevant_retrieval (FP) predicted practice not in ground truth
    - formatting_failure   JSON parse / LLM output errors
    - no_evidence          retriever returned zero chunks

Usage (from project root, .venv activated):
    python scripts/04_error_analysis.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

RESULTS_DIR = Path("results")
CONFIGS = ["fixed_300", "fixed_500", "semantic"]


def load_per_policy(config: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / f"{config}_per_policy.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_errors(config: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / f"{config}_errors.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def categorise_errors(per_policy: pd.DataFrame, errors: pd.DataFrame | None, config: str) -> list[dict]:
    """Produce one row per policy with error categorisation."""
    rows = []
    for _, row in per_policy.iterrows():
        pid = row["policy_id"]
        tp, fp, fn = int(row["tp"]), int(row["fp"]), int(row["fn"])

        if tp == 0 and fp == 0 and fn == 0:
            category = "no_predictions_no_gt"
        elif fn > 0 and fp == 0:
            category = "missed_risk_only"
        elif fp > 0 and fn == 0:
            category = "irrelevant_retrieval_only"
        elif fn > 0 and fp > 0:
            category = "mixed_errors"
        elif tp > 0 and fn == 0 and fp == 0:
            category = "perfect_match"
        else:
            category = "other"

        rows.append({
            "config": config,
            "policy_id": pid,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(row["precision"], 3),
            "recall": round(row["recall"], 3),
            "f1": round(row["f1"], 3),
            "error_category": category,
        })

    if errors is not None:
        for _, erow in errors.iterrows():
            pid = erow["policy_id"]
            error_text = str(erow.get("error", ""))
            if "no_evidence" in error_text.lower() or "no indexed" in error_text.lower():
                cat = "no_evidence_retrieved"
            elif "json" in error_text.lower() or "no json" in error_text.lower():
                cat = "formatting_failure"
            elif "timeout" in error_text.lower():
                cat = "llm_timeout"
            else:
                cat = "formatting_failure"

            rows.append({
                "config": config,
                "policy_id": pid,
                "tp": 0, "fp": 0, "fn": 0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "error_category": cat,
            })

    return rows


def generate_summary_md(all_rows: pd.DataFrame, output_path: Path) -> None:
    """Generate a markdown summary of computed results — tables only, no
    pre-written interpretive conclusions. Analysis and discussion of
    what these numbers mean belongs in the dissertation, written by the
    author after reviewing the actual tables below."""
    lines = [
        "# Error Analysis Summary\n",
        "Auto-generated from experiment results (real pipeline output — not illustrative).",
        "Tables only; analysis and discussion belongs in the dissertation, written after",
        "reviewing these numbers.\n",
    ]

    for config in CONFIGS:
        cfg_data = all_rows[all_rows["config"] == config]
        if cfg_data.empty:
            continue

        lines.append(f"\n## {config}\n")

        cat_counts = cfg_data["error_category"].value_counts().to_dict()
        total = len(cfg_data)
        lines.append(f"Total policies processed: **{total}**\n")
        lines.append("| Error Category | Count | % |")
        lines.append("|---|---|---|")
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            pct = round(100 * count / total, 1)
            lines.append(f"| {cat} | {count} | {pct}% |")

        evaluated = cfg_data[~cfg_data["error_category"].isin(
            ["formatting_failure", "no_evidence_retrieved", "llm_timeout"]
        )]
        if not evaluated.empty:
            fn_total = evaluated["fn"].sum()
            fp_total = evaluated["fp"].sum()
            tp_total = evaluated["tp"].sum()

            lines.append(f"\n### Missed vs Irrelevant\n")
            lines.append(f"- **Total True Positives**: {tp_total}")
            lines.append(f"- **Total False Negatives (missed risks)**: {fn_total}")
            lines.append(f"- **Total False Positives (irrelevant/hallucinated)**: {fp_total}")

            if fn_total + fp_total > 0:
                fn_pct = round(100 * fn_total / (fn_total + fp_total), 1)
                fp_pct = round(100 * fp_total / (fn_total + fp_total), 1)
                lines.append(f"- Of all errors, **{fn_pct}%** were missed risks and **{fp_pct}%** were irrelevant predictions.")

        if not evaluated.empty:
            worst = evaluated.nsmallest(5, "f1")
            lines.append(f"\n### Worst-Performing Policies (by F1)\n")
            lines.append("| Policy ID | Precision | Recall | F1 | TP | FP | FN |")
            lines.append("|---|---|---|---|---|---|---|")
            for _, r in worst.iterrows():
                lines.append(f"| {r['policy_id']} | {r['precision']} | {r['recall']} | {r['f1']} | {r['tp']} | {r['fp']} | {r['fn']} |")

        if not evaluated.empty:
            best = evaluated.nlargest(5, "f1")
            lines.append(f"\n### Best-Performing Policies (by F1)\n")
            lines.append("| Policy ID | Precision | Recall | F1 | TP | FP | FN |")
            lines.append("|---|---|---|---|---|---|---|")
            for _, r in best.iterrows():
                lines.append(f"| {r['policy_id']} | {r['precision']} | {r['recall']} | {r['f1']} | {r['tp']} | {r['fp']} | {r['fn']} |")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written summary to {output_path}")


def main():
    all_rows = []

    for config in CONFIGS:
        per_policy = load_per_policy(config)
        if per_policy is None:
            print(f"Skipping {config}: no per_policy CSV found")
            continue

        errors = load_errors(config)
        rows = categorise_errors(per_policy, errors, config)
        all_rows.extend(rows)
        print(f"{config}: {len(rows)} policies categorised")

    if not all_rows:
        print("No data found. Run scripts/02_run_experiment.py first.")
        return

    df = pd.DataFrame(all_rows)

    csv_path = RESULTS_DIR / "error_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWritten {len(df)} rows to {csv_path}")

    print("\n=== Error Category Summary ===")
    summary = df.groupby(["config", "error_category"]).size().reset_index(name="count")
    print(summary.to_string(index=False))

    md_path = RESULTS_DIR / "error_analysis_summary.md"
    generate_summary_md(df, md_path)


if __name__ == "__main__":
    main()