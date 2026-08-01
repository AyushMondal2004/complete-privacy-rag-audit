from __future__ import annotations
import os
import chromadb
from dotenv import load_dotenv
load_dotenv()
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/processed/chroma")
_client = None
def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=PERSIST_DIR)
    return _client
def get_or_create_collection(config_name: str) -> chromadb.Collection:
    client = get_client()
    # Use cosine distance for similarity matching
    return client.get_or_create_collection(
        name=config_name,
        metadata={"hnsw:space": "cosine"}
    )
def add_chunks(collection: chromadb.Collection, chunks: list, embeddings: list) -> None:
    if not chunks:
        return
    ids = []
    documents = []
    metadatas = []
    for c in chunks:
        ids.append(c.chunk_id)
        documents.append(c.text)
        meta = {
            "policy_id": c.policy_id,
            "position": c.position,
            "config_name": c.config_name,
        }
        if c.section_heading is not None:
            meta["section_heading"] = c.section_heading
        if c.metadata:
            for k, v in c.metadata.items():
                if v is not None and type(v) in (str, int, float, bool):
                    meta[k] = v                    
        metadatas.append(meta)
    batch_size = 5000
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
def query_policy(
    collection: chromadb.Collection, 
    query_embedding: list[float], 
    policy_id: str, 
    top_k: int = 10, 
    similarity_threshold: float | None = None
) -> list[dict]:
    embedding_list = query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)
    res = collection.query(
        query_embeddings=[embedding_list],
        n_results=top_k,
        where={"policy_id": policy_id}
    )
    evidence = []
    if not res.get("ids") or not res["ids"][0]:
        return evidence
    for i in range(len(res["ids"][0])):
        dist = res["distances"][0][i]
        sim = 1.0 - dist
        if similarity_threshold is not None and sim < similarity_threshold:
            continue
        meta = res["metadatas"][0][i] if (res.get("metadatas") and res["metadatas"][0]) else {}
        evidence.append({
            "chunk_id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "similarity": sim,
            "metadata": meta
        })
    return evidence