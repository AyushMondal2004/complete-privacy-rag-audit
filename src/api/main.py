from __future__ import annotations
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.ingestion.html_cleaner import clean_html
from src.ingestion.loader import load_ground_truth
from src.chunking.fixed_chunker import chunk_fixed
from src.vectorstore.chroma_store import get_or_create_collection, add_chunks
from src.embeddings.embedder import embed_texts
from src.retrieval.retriever import retrieve_evidence
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.generation.llm_client import call_llm, parse_llm_json
from src.evaluation.label_matcher import extract_ground_truth_labels, extract_predicted_labels, match_labels
from src.evaluation.metrics import compute_metrics
app = FastAPI(title="Privacy-RAG-Audit")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
class AuditRequest(BaseModel):
    config_name: str = "fixed_300"
    top_k: int = 10
    similarity_threshold: float | None = None
class AuditResponse(BaseModel):
    policy_id: str
    predicted: dict
    evidence_used: list[dict]
    metrics: dict | None = None
@app.get("/health")
def health():
    return {"status": "ok"}
@app.post("/experiment/{policy_id}", response_model=AuditResponse)
def run_experiment(policy_id: str, req: AuditRequest):
    collection = get_or_create_collection(req.config_name)
    evidence = retrieve_evidence(collection, policy_id, top_k=req.top_k, similarity_threshold=req.similarity_threshold)
    if not evidence:
        raise HTTPException(404, f"No indexed chunks found for policy_id={policy_id} in config={req.config_name}. "
                                  f"Run scripts/01_build_vector_db.py first.")
    user_prompt = build_user_prompt(policy_id, evidence)
    raw = call_llm(SYSTEM_PROMPT, user_prompt)
    predicted_json = parse_llm_json(raw)
    metrics_dict = None
    gt = load_ground_truth(policy_id)
    if gt is not None:
        predicted_labels = extract_predicted_labels(predicted_json)
        gt_labels = extract_ground_truth_labels(gt)
        match = match_labels(policy_id, predicted_labels, gt_labels)
        m = compute_metrics(len(match.true_positives), len(match.false_positives), len(match.false_negatives))
        metrics_dict = {
            "precision": round(m.precision, 3),
            "recall": round(m.recall, 3),
            "f1": round(m.f1, 3),
            "true_positives": match.true_positives,
            "false_positives": match.false_positives,
            "false_negatives": match.false_negatives,
        }
    return AuditResponse(policy_id=policy_id, predicted=predicted_json, evidence_used=evidence, metrics=metrics_dict)
@app.post("/audit")
async def audit_uploaded_policy(file: UploadFile, top_k: int = 10):
    raw_html = (await file.read()).decode("utf-8", errors="ignore")
    cleaned = clean_html(policy_id="uploaded", raw_html=raw_html)
    chunks = chunk_fixed("uploaded", cleaned.text, chunk_size_words=300, overlap_words=50, config_name="ad_hoc")
    if not chunks:
        raise HTTPException(400, "Could not extract usable text from the uploaded document.")
    embeddings = embed_texts([c.text for c in chunks])
    from src.vectorstore.chroma_store import get_client
    client = get_client()
    collection = client.get_or_create_collection(name="ad_hoc_uploads")
    add_chunks(collection, chunks, embeddings)
    evidence = retrieve_evidence(collection, "uploaded", top_k=top_k)
    user_prompt = build_user_prompt("uploaded", evidence)
    raw = call_llm(SYSTEM_PROMPT, user_prompt)
    predicted_json = parse_llm_json(raw)
    client.delete_collection("ad_hoc_uploads")
    return {"predicted": predicted_json, "evidence_used": evidence}