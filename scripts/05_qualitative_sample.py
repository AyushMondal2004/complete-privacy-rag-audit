from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
import pandas as pd
from tqdm import tqdm
from src.ingestion.loader import iter_raw_policies
from src.vectorstore.chroma_store import get_or_create_collection
from src.retrieval.retriever import retrieve_evidence
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.generation.llm_client import call_llm, parse_llm_json
OUT_DIR = Path("results/qualitative_review")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n", type=int, default=25, help="Number of policies to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, for a reproducible sample")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    collection = get_or_create_collection(cfg["config_name"])
    all_policies = list(iter_raw_policies())
    random.seed(args.seed)
    sample = random.sample(all_policies, min(args.n, len(all_policies)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_results = []
    review_rows = []
    for raw_policy in tqdm(sample, desc=f"Sampling {cfg['config_name']}"):
        pid = raw_policy.policy_id
        try:
            evidence = retrieve_evidence(
                collection, pid,
                top_k=cfg["top_k"],
                similarity_threshold=cfg["similarity_threshold"],
            )
            if not evidence:
                continue
            user_prompt = build_user_prompt(pid, evidence)
            raw = call_llm(SYSTEM_PROMPT, user_prompt)
            predicted_json = parse_llm_json(raw)
            raw_results.append({"policy_id": pid, "predicted": predicted_json})
            for p in predicted_json.get("practices", []):
                review_rows.append({
                    "policy_id": pid,
                    "practice": p.get("practice", ""),
                    "modality": p.get("modality", ""),
                    "evidence": p.get("evidence", ""),
                    "explanation": p.get("explanation", ""),
                    "grounded_in_evidence": "",
                    "understandable": "",
                    "not_misleading": "",
                    "reviewer_notes": "",
                })
        except Exception as e:
            print(f"[skip] {pid}: {e}")
    with open(OUT_DIR / f"{cfg['config_name']}_sample_raw.json", "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2)
    review_df = pd.DataFrame(review_rows)
    review_df.to_csv(OUT_DIR / f"{cfg['config_name']}_review_sheet.csv", index=False)
    print(f"\nSampled {len(raw_results)} policies, {len(review_rows)} individual practice predictions.")
    print(f"Raw JSON: {OUT_DIR / (cfg['config_name'] + '_sample_raw.json')}")
    print(f"Review sheet (fill in by hand): {OUT_DIR / (cfg['config_name'] + '_review_sheet.csv')}")
if __name__ == "__main__":
    main()