# Architecture notes

Reference notes for writing your own Chapter 3 (Methodology) and Chapter 4
(Results) — not dissertation prose. Use these as a technical map; write the
narrative yourself in your own words with your own actual numbers.

This file reflects the pipeline as it actually ran for the full
349-policy experiments. See `docs/limitations.md` for issues found
during development and what remains unresolved.

## Component map

| Layer | Module | Responsibility |
|---|---|---|
| Ingestion | `src/ingestion/html_cleaner.py` | Strip boilerplate HTML, preserve headings, normalise whitespace |
| Ingestion | `src/ingestion/loader.py` | Walk APP-350 corpus (`policy_<id>.yml` annotations), pair HTML documents with YAML annotations |
| Chunking | `src/chunking/fixed_chunker.py` | Baseline: fixed word count + overlap (300 or 500 words) |
| Chunking | `src/chunking/structural_chunker.py` | Heading-aware section splitting — built but not used in the three-way comparison |
| Chunking | `src/chunking/semantic_chunker.py` | Sentence-embedding similarity breakpoints — the best-performing config |
| Embedding | `src/embeddings/embedder.py` | Sentence-transformers wrapper (default: all-MiniLM-L6-v2) |
| Storage | `src/vectorstore/chroma_store.py` | One Chroma collection per config, policy_id metadata filter, chromadb 0.5.5 (see note below) |
| Retrieval | `src/retrieval/retriever.py` | Multi-query retrieval: 6 hand-written category queries, merged and de-duplicated by chunk_id |
| Generation | `src/generation/prompts.py`, `llm_client.py` | One structured-JSON extraction call per policy, local LLM via Ollama (qwen2.5:7b for all reported runs) |
| Evaluation | `src/evaluation/label_matcher.py`, `metrics.py` | Predicted vs ground-truth label matching (segment-based YAML schema), precision/recall/F1 |
| Error analysis | `scripts/04_error_analysis.py` | Policy-level error categorisation (missed_risk_only / irrelevant_retrieval_only / mixed_errors / perfect_match) |
| Visualisation | `notebooks/01_results_visualisation.py` | Generates the 5 figures in `results/figures/` |
| API | `src/api/main.py` | FastAPI endpoints: `/experiment/{policy_id}` (with ground truth), `/audit` (ad hoc upload). CORS enabled for the local frontend. |
| Frontend | `frontend/index.html` | Single-page audit UI: policy lookup or file upload, risk-severity badges, evidence display, ground-truth comparison |

## Retrieval design: multi-query, not single generic query

An earlier design used one generic query ("identify all privacy
practices..."). Debugging showed this consistently under-retrieved —
for a policy with a Bluetooth permission clause, the retriever never
surfaced it regardless of `top_k`, because the generic query sits
semantically far from specific device-permission phrasing.

The retriever now runs 6 targeted category queries (contact, location,
device identifiers, demographics, third-party sharing, SSO/login),
merges results by chunk_id, and keeps the highest similarity score seen
per chunk. `top_k` in the retriever config is per-query, not overall.

This did not fully solve the problem — further debugging showed that
even when the relevant evidence chunk was retrieved and passed to the
LLM, the model sometimes still failed to extract from it (see
`docs/limitations.md`, point 1). The retrieval fix increased evidence
coverage; it did not guarantee the LLM would use all of it.

## Why one comprehensive prompt instead of eight per-category questions

Querying once per privacy-practice category multiplies LLM calls by ~8x
for no retrieval benefit, since the same evidence chunks are usually
relevant to several categories at once. The single structured-JSON
extraction prompt (`src/generation/prompts.py`) asks the model to
return every supported practice in one pass, using a fixed vocabulary
of APP-350-style labels stated explicitly in the system prompt (this
vocabulary was necessary — without it, the model invented its own label
names that never matched ground truth).

## Retrieval/generation parameters used in the full 349-policy runs

All three configs used identical retrieval settings to isolate chunking
strategy as the controlled variable:
- `top_k: 8` (per category query)
- `similarity_threshold: 0.1`
- LLM timeout: 420s
- LLM temperature: 0.0
- Model: qwen2.5:7b via Ollama

These values were chosen via 20–30 policy pilot runs before the full
run, and separately validated by a threshold sweep
(`experiments/threshold_sweep/`) on a 15-policy subsample per config,
which showed thresholds 0.0–0.2 produced near-identical results (the
filter wasn't removing meaningful content at those levels).

## Known environment issue: chromadb version upgrade

`chromadb` was upgraded from 0.4.24 to 0.5.5 partway through the
project. This broke the previously stored vector database (collection
metadata format changed) and required two fixes:
1. Deleting and rebuilding `data/processed/chroma/` from scratch.
2. A code fix in `chroma_store.py`'s `query_policy()` — 0.5.5 rejects
   numpy arrays in `query_embeddings`, requiring an explicit
   `.tolist()` conversion.

`requirements.txt` pins `chromadb==0.5.5`; any future re-run should use
this exact version to avoid the same issue recurring.

## Results summary (full 349-policy runs)

| Config | Precision | Recall | F1 | Pipeline errors |
|---|---|---|---|---|
| fixed_300 | 0.440 | 0.102 | 0.166 | 8 (5 JSON, 3 timeout) |
| fixed_500 | 0.457 | 0.100 | 0.164 | 7 (4 JSON, 3 timeout) |
| semantic | 0.496 | 0.258 | 0.339 | 13 (4 JSON, 9 timeout) |

See `results/comparison_summary.csv`, `results/error_analysis_summary.md`,
and `results/figures/` for the full breakdown and charts. See
`docs/limitations.md` for what these numbers do and don't support.