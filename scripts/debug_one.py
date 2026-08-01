from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from src.ingestion.loader import load_ground_truth
from src.vectorstore.chroma_store import get_or_create_collection
from src.retrieval.retriever import retrieve_evidence
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.generation.llm_client import call_llm, parse_llm_json
from src.evaluation.label_matcher import extract_ground_truth_labels, extract_predicted_labels, normalise_label
policy_id = sys.argv[1] if len(sys.argv) > 1 else "108"
with open("experiments/configs/fixed_300.yaml") as f:
    cfg = yaml.safe_load(f)
collection = get_or_create_collection(cfg["config_name"])
evidence = retrieve_evidence(collection, policy_id, top_k=cfg["top_k"], similarity_threshold=cfg["similarity_threshold"])
print(f"=== Retrieved {len(evidence)} chunks for policy {policy_id} ===")
for e in evidence:
    print(f"  {e['metadata'].get('position')} | sim={e['similarity']:.2f} | {e['text'][:100]}...")
print()
user_prompt = build_user_prompt(policy_id, evidence)
raw = call_llm(SYSTEM_PROMPT, user_prompt)
print("\n=== RAW LLM OUTPUT ===")
print(raw)
predicted_json = parse_llm_json(raw)
predicted_labels = extract_predicted_labels(predicted_json)
gt = load_ground_truth(policy_id)
gt_labels = extract_ground_truth_labels(gt) if gt else set()
print("\n=== PREDICTED LABELS (normalised) ===")
print(sorted(predicted_labels))
print("\n=== GROUND TRUTH LABELS (normalised) ===")
print(sorted(gt_labels))
print("\n=== OVERLAP ===")
print("Matched:", sorted(predicted_labels & gt_labels))
print("Predicted but not in GT:", sorted(predicted_labels - gt_labels))
print("In GT but not predicted:", sorted(gt_labels - predicted_labels))