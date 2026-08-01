from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from tqdm import tqdm
from src.ingestion.loader import iter_raw_policies
from src.ingestion.html_cleaner import clean_html
from src.chunking.fixed_chunker import chunk_fixed
from src.chunking.structural_chunker import chunk_structural
from src.chunking.semantic_chunker import chunk_semantic
from src.embeddings.embedder import embed_texts
from src.vectorstore.chroma_store import get_or_create_collection, add_chunks
def build_chunks(cfg: dict, policy_id: str, text: str):
    if cfg["strategy"] == "fixed":
        return chunk_fixed(
            policy_id, text,
            chunk_size_words=cfg["chunk_size_words"],
            overlap_words=cfg["overlap_words"],
            config_name=cfg["config_name"],
        )
    elif cfg["strategy"] == "structural":
        return chunk_structural(policy_id, text, config_name=cfg["config_name"])
    elif cfg["strategy"] == "semantic":
        return chunk_semantic(
            policy_id, text,
            embed_fn=lambda texts: embed_texts(texts),
            similarity_threshold=cfg["similarity_threshold_chunking"],
            min_chunk_words=cfg["min_chunk_words"],
            max_chunk_words=cfg["max_chunk_words"],
            config_name=cfg["config_name"],
        )
    else:
        raise ValueError(f"Unknown strategy: {cfg['strategy']}")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of policies (useful for a quick pilot run before the full 350)")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    collection = get_or_create_collection(cfg["config_name"])
    policies = list(iter_raw_policies())
    if not policies:
        print("No policies found under data/raw/ — copy the APP-350 corpus there first "
              "(see README + src/ingestion/loader.py for the expected layout).")
        return
    if args.limit:
        policies = policies[: args.limit]
    n_indexed, n_skipped = 0, 0
    for raw_policy in tqdm(policies, desc=f"Indexing ({cfg['config_name']})"):
        try:
            raw_html = raw_policy.html_path.read_text(encoding="utf-8", errors="ignore")
            cleaned = clean_html(raw_policy.policy_id, raw_html)
            chunks = build_chunks(cfg, raw_policy.policy_id, cleaned.text)
            if not chunks:
                n_skipped += 1
                continue
            embeddings = embed_texts([c.text for c in chunks])
            add_chunks(collection, chunks, embeddings)
            n_indexed += 1
        except Exception as e:
            print(f"[skip] {raw_policy.policy_id}: {e}")
            n_skipped += 1
    print(f"Done. Indexed {n_indexed} policies, skipped {n_skipped}, config={cfg['config_name']}.")
if __name__ == "__main__":
    main()