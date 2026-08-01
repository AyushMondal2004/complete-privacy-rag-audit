from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
import pandas as pd
from tqdm import tqdm
from src.ingestion.loader import iter_raw_policies, load_ground_truth
from src.vectorstore.chroma_store import get_or_create_collection
from src.retrieval.retriever import retrieve_evidence
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.generation.llm_client import call_llm, parse_llm_json
from src.evaluation.label_matcher import extract_ground_truth_labels, extract_predicted_labels, match_labels
from src.evaluation.metrics import aggregate, per_policy_metrics
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    collection = get_or_create_collection(cfg["config_name"])
    policies = list(iter_raw_policies())
    if args.limit:
        policies = policies[: args.limit]
    match_results = []
    error_log = []
    for raw_policy in tqdm(policies, desc=f"Running experiment ({cfg['config_name']})"):
        pid = raw_policy.policy_id
        gt = load_ground_truth(pid)
        if gt is None:
            continue
        try:
            evidence = retrieve_evidence(
                collection, pid,
                top_k=cfg["top_k"],
                similarity_threshold=cfg["similarity_threshold"],
            )
            if not evidence:
                error_log.append({"policy_id": pid, "error": "no_evidence_retrieved"})
                continue
            user_prompt = build_user_prompt(pid, evidence)
            raw = call_llm(SYSTEM_PROMPT, user_prompt)
            predicted_json = parse_llm_json(raw)
            predicted_labels = extract_predicted_labels(predicted_json)
            gt_labels = extract_ground_truth_labels(gt)
            match_results.append(match_labels(pid, predicted_labels, gt_labels))
        except Exception as e:
            error_log.append({"policy_id": pid, "error": str(e)})
        time.sleep(0.1)
    if not match_results:
        print("No policies were successfully evaluated. Check data/raw/ layout and that your LLM server is running.")
        if error_log:
            print("First 5 errors:")
            for err in error_log[:5]:
                print(err)
        return
    Path("results").mkdir(exist_ok=True)
    per_policy = per_policy_metrics(match_results)
    pd.DataFrame(per_policy).to_csv(f"results/{cfg['config_name']}_per_policy.csv", index=False)
    if error_log:
        pd.DataFrame(error_log).to_csv(f"results/{cfg['config_name']}_errors.csv", index=False)
    overall = aggregate(match_results)
    summary = {
        "config_name": cfg["config_name"],
        "n_policies_evaluated": len(match_results),
        "n_errors": len(error_log),
        "precision": round(overall.precision, 3),
        "recall": round(overall.recall, 3),
        "f1": round(overall.f1, 3),
    }
    pd.DataFrame([summary]).to_csv(f"results/{cfg['config_name']}_summary.csv", index=False)
    print("\n=== Result ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
if __name__ == "__main__":
    main()