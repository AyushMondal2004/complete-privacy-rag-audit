from __future__ import annotations
from src.embeddings.embedder import embed_texts
from src.vectorstore.chroma_store import query_policy

# Multiple targeted queries instead of one generic query. A single
# generic query ("identify all privacy practices...") sits semantically
# far from specific sections about e.g. Bluetooth or ad IDs, so it never
# retrieves them even with a high top_k. Querying per category and
# merging results covers far more of the policy's actual practice
# statements. See docs/architecture.md for the reasoning and the debug
# trace that motivated this change.
CATEGORY_QUERIES = [
    "What personal contact information does this app collect, such as email, phone number, or address?",
    "Does this app collect the user's precise or approximate location, GPS, WiFi, cell tower, or Bluetooth data?",
    "What device identifiers does this app collect, such as device ID, advertising ID, IP address, MAC address, or cookies?",
    "Does this app collect demographic information such as age or gender?",
    "Does this app share, sell, or disclose user information with third parties or partners?",
    "Does this app use single sign-on, Facebook login, or other third-party account login?",
]


def retrieve_evidence(
    collection,
    policy_id: str,
    top_k: int = 6,
    similarity_threshold: float | None = None,
    queries: list[str] | None = None,
):
    """Run each query in `queries` against the policy's chunks, merge and
    de-duplicate by chunk_id, keeping the highest similarity score seen
    for each chunk. top_k here is PER QUERY, not overall — six queries
    at top_k=6 can surface up to 36 distinct chunks, which is what
    actually raises recall (see debug_one.py output for policy 108,
    where a single top_k=25 query only returned 8 results)."""
    query_list = queries or CATEGORY_QUERIES
    query_embeddings = embed_texts(query_list)

    merged: dict[str, dict] = {}
    for query_embedding in query_embeddings:
        hits = query_policy(collection, query_embedding, policy_id, top_k, similarity_threshold)
        for hit in hits:
            cid = hit["chunk_id"]
            if cid not in merged or hit["similarity"] > merged[cid]["similarity"]:
                merged[cid] = hit

    return sorted(merged.values(), key=lambda h: h["similarity"], reverse=True)