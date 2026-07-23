# Limitations

This document consolidates limitations actually observed while building and
running this pipeline. Each point below is grounded in a specific run,
error log, or debugging session — not a generic disclaimer list. Use this
as reference material for your dissertation's Limitations section; the
prose interpretation and framing should be written in your own words.

## 1. Local LLM (qwen2.5:7b) extraction is incomplete, not just noisy

Debugging a single policy (id 108) directly against the raw retrieved
evidence showed the model reading only 3–4 of 8 retrieved excerpts before
stopping, despite an explicit "consider every excerpt" instruction in the
system prompt. This is visible in `scripts/debug_one.py` output saved
during the session: evidence chunk `[4]` (containing a Bluetooth
permission description) was present in every retrieval call across
several test runs but was never extracted, even after strengthening the
prompt.

This is consistent with the aggregate error-category breakdown in
`results/error_analysis_summary.md`: `missed_risk_only` and
`mixed_errors` together account for the large majority of policies
across all three configs (89.4% for fixed_300, 88.0% for fixed_500,
87.4% for semantic), and total false negatives (3,340 / 3,371 / 2,638)
substantially exceed total false positives (482 / 446 / 934) in every
configuration. The model under-predicts more than it over-predicts.

## 2. Citation/attribution drift under long structured input

In the same debugging session, the model was observed citing the same
evidence excerpt number for multiple unrelated predicted practices (e.g.
citing a "read your contacts" excerpt as evidence for GPS, IP address,
cell tower, and network-based location simultaneously). This suggests
the model pattern-matches on the *structure* of permission-list text
(itemised, repetitive phrasing) rather than reading each item's specific
content when the input is long. This was not systematically quantified
across all 349 policies — it was observed on a small number of manually
inspected cases and should be described as an illustrative finding, not
a measured rate, unless further sampling is done.

## 3. Party attribution (1st-party vs 3rd-party) errors

During debugging, the model mislabelled a first-party action (the app
itself reading an ad ID) as `_3rdParty`, apparently based on the topic
(advertising) rather than the actor performing the action. A prompt
revision explicitly defining "who performs the action, not who benefits"
was added mid-session; its effect on the full 349-policy run was not
separately re-measured against a pre-revision baseline, since the fix
was made before the final full runs. This limitation is about a possible
residual failure mode, not a measured one.

## 4. Retrieval used a fixed, hand-authored category-query set

`src/retrieval/retriever.py` retrieves evidence using six manually
written category queries (contact, location, identifiers, demographics,
third-party sharing, SSO) rather than a single generic query. This
change was made after observing that a single generic query failed to
retrieve evidence for several ground-truth categories (see the
`docs/architecture.md` note on this). The six categories were chosen to
match the APP-350 practice taxonomy at a coarse level; they do not
individually cover every fine-grained practice label in that taxonomy
(e.g. `Contact_ZIP` and `Contact_1stParty` are not separately queried).

## 5. Retrieval and LLM-extraction parameters were tuned on a small pilot, then applied uniformly

`top_k` and `similarity_threshold` were selected via 20–30 policy pilot
runs, then held constant across all three chunking configurations for
the full 349-policy run, to isolate chunking strategy as the controlled
variable (see `docs/architecture.md`). This means retrieval depth was
not separately optimised per chunking strategy — semantic chunking's
variable-length chunks may not be equally well served by the same
`top_k` as fixed-length chunks. The threshold sweep
(`experiments/threshold_sweep/`) explored threshold sensitivity on a
15-policy subsample per configuration, not the full corpus.

## 6. Label matching relies on a normalisation/alias table with known gaps

`src/evaluation/label_matcher.py` normalises both predicted and
ground-truth labels and maps a small number of observed near-miss label
variants (e.g. `Identifier_Advertising` → `Identifier_Ad_ID`) based on
patterns seen during debugging of a handful of policies. This alias
table is not exhaustive — it was built reactively from cases actually
observed, not from a systematic review of every possible label variant
the LLM could produce. Some genuine matches may still be scored as
mismatches if the model produces a label variant not covered by the
alias table.

## 7. Category-level (RQ5) results are based on a 15-policy pilot, not the full corpus

The category heatmap (`results/figures/category_heatmap.png`) reports
F1 by privacy-practice category and chunking configuration, but this
was computed on a 15-policy pilot subsample, not the full 349-policy
corpus, because per-label match results were not logged during the main
experiment runs (only aggregate TP/FP/FN counts per policy were saved).
Category-level conclusions should be treated as indicative rather than
definitive, and this should be stated explicitly wherever the heatmap is
referenced.

## 8. Pipeline error rates differ meaningfully by configuration

Across the full 349-policy runs: fixed_300 had 8 pipeline errors (5
JSON parse failures, 3 LLM timeouts), fixed_500 had 7 (4 JSON, 3
timeout), and semantic had 13 (4 JSON, 9 timeout) — see
`results/*_errors.csv`. Semantic chunking's higher timeout rate is
plausibly linked to longer per-policy LLM calls (average latency
~47s/policy vs ~32s/policy for the fixed configs, observed directly
from the terminal output during each full run), itself likely a
consequence of semantic chunking producing more, and more granular,
retrieved evidence per query. This is a real compute-cost trade-off
alongside the accuracy improvement, not a cost-free win.

## 9. Software environment change mid-project

`chromadb` was upgraded from 0.4.24 to 0.5.5 partway through the
project (via `pip install -r requirements.txt` in a later session),
which broke backward compatibility with the previously stored vector
database format (`KeyError: '_type'` when loading old collections) and
separately surfaced a stricter input-type check that required a code
fix (`ValueError: Expected each embedding... to be a list, got
['ndarray']`) in `src/vectorstore/chroma_store.py`. All vector
databases were rebuilt from scratch after this upgrade, so reported
results are consistent with the current dependency versions pinned in
`requirements.txt` — but this is a reminder that dependency pinning
matters for reproducibility, and any future re-run should use the exact
versions in that file.

## 10. Single local model, no cross-model comparison in the full results

All 349-policy experiment runs used `qwen2.5:7b`. A pilot comparison
against `llama3.1:8b` was planned but not completed within the project
timeline (see project history) — the `.env`-based model switch did not
take effect in one attempted comparison run due to a terminal
environment-variable loading issue, and the comparison was not
re-attempted before the deadline. Findings about extraction quality,
citation drift, and timeout behaviour are specific to qwen2.5:7b and
should not be assumed to generalise to other locally-run models without
further testing.

## 11. Not a substitute for legal or compliance review

As stated in the project proposal: this system is a decision-support
tool that summarises and highlights possible privacy practices based on
retrieved policy text. It does not provide legal advice, and given the
recall limitations documented above, it should not be treated as an
exhaustive or authoritative audit of any policy's actual practices.