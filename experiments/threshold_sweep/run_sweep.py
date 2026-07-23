"""
Threshold + Config Sensitivity Pilot Study
==========================================
Runs the full RAG pipeline on a randomly sampled subset of 30 policies
across ALL THREE chunking configs and ALL FIVE cosine-similarity thresholds:

    Configs     : fixed_300 | fixed_500 | semantic
    Thresholds  : 0.0 | 0.1 | 0.2 | 0.3 | 0.4

The SAME 10 policies are used for every config/threshold combination so
results are directly comparable (controlled experiment).

Output structure (all inside this folder - main results/ is never touched):
    results/
        fixed_300/
            threshold_0_0.csv   <- per-policy rows at threshold=0.0
            threshold_0_1.csv
            ...
            summary.csv         <- aggregated P/R/F1 per threshold
        fixed_500/
            threshold_0_0.csv
            ...
            summary.csv
        semantic/
            threshold_0_0.csv
            ...
            summary.csv
        all_configs_summary.csv <- combined table for thesis (all configs x thresholds)

Reproducibility:
    Fixed random seed (42). Sampled policy IDs saved to sampled_policy_ids.txt.
    Include this file in your thesis appendix.

Usage (from project root, .venv activated):
    python experiments/threshold_sweep/run_sweep.py

Options:
    --n 10                            number of policies to sample (default: 30)
    --configs fixed_300 fixed_500 semantic   configs to include (default: all three)
    --thresholds 0.0 0.1 0.2 0.3 0.4        thresholds to sweep (default: all five)
    --resample                        force a new random sample

Prerequisites:
    All three ChromaDB collections must already be indexed:
        python scripts/01_build_vector_db.py --config experiments/configs/fixed_300.yaml
        python scripts/01_build_vector_db.py --config experiments/configs/fixed_500.yaml
        python scripts/01_build_vector_db.py --config experiments/configs/semantic.yaml
    Ollama LLM server must be running.

Estimated runtime:
    3 configs x 5 thresholds x 10 policies x ~10s/policy = ~4.5 hours total.
    Each threshold/config result is saved immediately (crash-safe).
    You can re-run and it will skip configs/thresholds already completed
    (unless --force is passed).
"""
from __future__ import annotations
import argparse
import random
import sys
import time
from pathlib import Path

# -- make project root importable regardless of cwd --------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from src.ingestion.loader import iter_raw_policies, load_ground_truth
from src.vectorstore.chroma_store import get_or_create_collection
from src.retrieval.retriever import retrieve_evidence
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.generation.llm_client import call_llm, parse_llm_json
from src.evaluation.label_matcher import (
    extract_ground_truth_labels,
    extract_predicted_labels,
    match_labels,
)

# -- constants ----------------------------------------------------------------
ALL_CONFIGS        = ["fixed_300", "fixed_500", "semantic"]
TOP_K              = 8                          # per-query top-k (matches main experiment)
RANDOM_SEED        = 42                         # fixed for reproducibility
DEFAULT_N          = 15
DEFAULT_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4]

THIS_DIR         = Path(__file__).parent
RESULTS_DIR      = THIS_DIR / "results"
POLICY_IDS_FILE  = THIS_DIR / "sampled_policy_ids.txt"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_policy_ids(n: int, raw_dir: Path) -> list[str]:
    """Randomly sample n policy IDs that have a ground-truth annotation.
    Saves to sampled_policy_ids.txt for thesis appendix reproducibility."""
    print("Scanning corpus for annotated policies...")
    all_ids = [
        p.policy_id
        for p in iter_raw_policies(raw_dir)
        if load_ground_truth(p.policy_id, raw_dir) is not None
    ]
    if not all_ids:
        print("[error] No annotated policies found in data/raw/. Check directory layout.")
        sys.exit(1)

    if len(all_ids) < n:
        print(f"[warn] Only {len(all_ids)} annotated policies available; using all.")
        n = len(all_ids)

    random.seed(RANDOM_SEED)
    sampled = sorted(random.sample(all_ids, n))
    POLICY_IDS_FILE.write_text("\n".join(sampled))
    print(f"Sampled {len(sampled)} policy IDs (seed={RANDOM_SEED}) -> {POLICY_IDS_FILE.name}")
    print(f"IDs: {', '.join(sampled)}\n")
    return sampled


def load_sampled_ids() -> list[str] | None:
    """Reuse an existing sample file so all configs use the identical 30 policies."""
    if POLICY_IDS_FILE.exists():
        ids = [l.strip() for l in POLICY_IDS_FILE.read_text().splitlines() if l.strip()]
        if ids:
            print(f"Reusing existing sample of {len(ids)} policy IDs from {POLICY_IDS_FILE.name}")
            return ids
    return None


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------
def run_one_threshold(
    collection,
    policy_ids: list[str],
    config_name: str,
    threshold: float,
    raw_dir: Path,
) -> tuple[list[dict], list[dict]]:
    """Run the full pipeline on all sampled policies at one threshold.
    Returns (per_policy_rows, error_rows)."""
    per_policy_rows: list[dict] = []
    error_rows: list[dict] = []

    for pid in tqdm(policy_ids, desc=f"    {config_name} | threshold={threshold:.1f}"):
        gt = load_ground_truth(pid, raw_dir)
        if gt is None:
            error_rows.append({"policy_id": pid, "error": "no_ground_truth"})
            continue
        try:
            evidence = retrieve_evidence(
                collection,
                pid,
                top_k=TOP_K,
                # None means no lower bound (retrieve everything)
                similarity_threshold=threshold if threshold > 0.0 else None,
            )
            if not evidence:
                error_rows.append({"policy_id": pid, "error": "no_evidence_retrieved"})
                continue

            user_prompt  = build_user_prompt(pid, evidence)
            raw_output   = call_llm(SYSTEM_PROMPT, user_prompt)
            predicted    = parse_llm_json(raw_output)

            pred_labels  = extract_predicted_labels(predicted)
            gt_labels    = extract_ground_truth_labels(gt)
            result       = match_labels(pid, pred_labels, gt_labels)

            tp = len(result.true_positives)
            fp = len(result.false_positives)
            fn = len(result.false_negatives)
            p  = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
            r  = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
            f1 = round(2 * p * r / (p + r), 3) if (p + r) > 0 else 0.0

            per_policy_rows.append({
                "policy_id":        pid,
                "config":           config_name,
                "threshold":        threshold,
                "n_chunks":         len(evidence),
                "precision":        p,
                "recall":           r,
                "f1":               f1,
                "tp":               tp,
                "fp":               fp,
                "fn":               fn,
                "true_positives":   "|".join(result.true_positives),
                "false_positives":  "|".join(result.false_positives),
                "false_negatives":  "|".join(result.false_negatives),
            })

        except Exception as exc:
            error_rows.append({"policy_id": pid, "error": str(exc)})

        time.sleep(0.1)  # gentle on the local LLM server

    return per_policy_rows, error_rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def make_summary(per_policy_rows: list[dict]) -> pd.DataFrame:
    """Micro-aggregate per-policy rows by threshold -> one summary row each."""
    df = pd.DataFrame(per_policy_rows)
    rows = []
    for threshold in sorted(df["threshold"].unique()):
        g  = df[df["threshold"] == threshold]
        tp = int(g["tp"].sum())
        fp = int(g["fp"].sum())
        fn = int(g["fn"].sum())
        p  = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
        r  = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
        f1 = round(2 * p * r / (p + r), 3) if (p + r) > 0 else 0.0
        rows.append({
            "config":     df["config"].iloc[0],
            "threshold":  threshold,
            "n_policies": len(g),
            "precision":  p,
            "recall":     r,
            "f1":         f1,
            "total_tp":   tp,
            "total_fp":   fp,
            "total_fn":   fn,
        })
    return pd.DataFrame(rows)


def thresh_filename(threshold: float) -> str:
    return "threshold_" + f"{threshold:.1f}".replace(".", "_") + ".csv"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Threshold + config sensitivity pilot study")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Policies to sample (default: {DEFAULT_N})")
    parser.add_argument("--configs", nargs="+", default=ALL_CONFIGS,
                        choices=ALL_CONFIGS,
                        help="Configs to run (default: all three)")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=DEFAULT_THRESHOLDS,
                        help="Thresholds to sweep (default: 0.0 0.1 0.2 0.3 0.4)")
    parser.add_argument("--resample", action="store_true",
                        help="Force a new random sample")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if output CSV already exists")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = PROJECT_ROOT / "data" / "raw"

    # -- sample policies (same IDs used for ALL configs) ----------------------
    if args.resample:
        policy_ids = sample_policy_ids(args.n, raw_dir)
    else:
        policy_ids = load_sampled_ids() or sample_policy_ids(args.n, raw_dir)

    print(f"\nWill sweep:")
    print(f"  Configs     : {args.configs}")
    print(f"  Thresholds  : {args.thresholds}")
    print(f"  Policies    : {len(policy_ids)}")
    total_runs = len(args.configs) * len(args.thresholds)
    print(f"  Total runs  : {total_runs} ({total_runs * len(policy_ids)} LLM calls)")
    print(f"  Est. time   : ~{total_runs * len(policy_ids) * 30 // 60} minutes\n")

    all_config_summaries: list[pd.DataFrame] = []

    for config_name in args.configs:
        print(f"\n{'#'*64}")
        print(f"  CONFIG: {config_name}")
        print(f"{'#'*64}")

        # verify collection exists
        collection = get_or_create_collection(config_name)
        n_chunks   = collection.count()
        if n_chunks == 0:
            print(f"[skip] ChromaDB collection '{config_name}' is empty.")
            print(f"       Run: python scripts/01_build_vector_db.py --config experiments/configs/{config_name}.yaml")
            continue
        print(f"  ChromaDB: {n_chunks:,} chunks indexed.")

        config_dir = RESULTS_DIR / config_name
        config_dir.mkdir(parents=True, exist_ok=True)

        config_rows: list[dict] = []

        for threshold in args.thresholds:
            out_path = config_dir / thresh_filename(threshold)

            # skip if already done (resume-friendly)
            if out_path.exists() and not args.force:
                print(f"\n  [skip] {out_path.name} already exists (use --force to re-run)")
                existing = pd.read_csv(out_path).to_dict("records")
                config_rows.extend(existing)
                continue

            print(f"\n  {'='*56}")
            print(f"  threshold = {threshold:.1f}  |  {len(policy_ids)} policies")
            print(f"  {'='*56}")

            rows, errs = run_one_threshold(
                collection, policy_ids, config_name, threshold, raw_dir
            )
            config_rows.extend(rows)

            # save per-threshold CSV immediately (crash-safe)
            pd.DataFrame(rows).to_csv(out_path, index=False)

            if errs:
                err_path = config_dir / (thresh_filename(threshold).replace(".csv", "_errors.csv"))
                pd.DataFrame(errs).to_csv(err_path, index=False)

            # inline mini-summary
            if rows:
                df_t = pd.DataFrame(rows)
                tp, fp, fn = df_t["tp"].sum(), df_t["fp"].sum(), df_t["fn"].sum()
                p  = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
                r  = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
                f1 = round(2 * p * r / (p + r), 3) if (p + r) > 0 else 0.0
                print(f"  Done: {len(rows)} OK, {len(errs)} errors | P={p:.3f}  R={r:.3f}  F1={f1:.3f}")

        # -- per-config summary.csv ------------------------------------------
        if config_rows:
            summary = make_summary(config_rows)
            summary_path = config_dir / "summary.csv"
            summary.to_csv(summary_path, index=False)
            print(f"\n  Summary saved -> results/{config_name}/summary.csv")
            print(summary.to_string(index=False))
            all_config_summaries.append(summary)

    # -- combined summary across all configs ----------------------------------
    if all_config_summaries:
        combined = pd.concat(all_config_summaries, ignore_index=True)
        combined = combined.sort_values(["config", "threshold"])
        combined_path = RESULTS_DIR / "all_configs_summary.csv"
        combined.to_csv(combined_path, index=False)

        print(f"\n\n{'='*64}")
        print("COMPLETE SWEEP SUMMARY  (all configs x thresholds)")
        print(f"{'='*64}")
        print(combined.to_string(index=False))
        print(f"\nSaved -> experiments/threshold_sweep/results/all_configs_summary.csv")


if __name__ == "__main__":
    main()
