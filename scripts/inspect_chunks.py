from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vectorstore.chroma_store import get_or_create_collection
policy_id = sys.argv[1] if len(sys.argv) > 1 else "108"
collection = get_or_create_collection("fixed_300")
print("Total chunks in fixed_300 collection (all policies):", collection.count())
result = collection.get(where={"policy_id": policy_id})
print(f"\nChunks stored for policy_id={policy_id}: {len(result['ids'])}")
for cid, doc in zip(result["ids"], result["documents"]):
    print(f"  {cid} | {len(doc.split())} words | {doc[:80]!r}")