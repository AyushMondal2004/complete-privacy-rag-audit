# Error Analysis Summary

Auto-generated from experiment results (real pipeline output — not illustrative).
Tables only; analysis and discussion belongs in the dissertation, written after
reviewing these numbers.


## fixed_300

Total policies processed: **349**

| Error Category | Count | % |
|---|---|---|
| mixed_errors | 161 | 46.1% |
| missed_risk_only | 151 | 43.3% |
| irrelevant_retrieval_only | 18 | 5.2% |
| perfect_match | 9 | 2.6% |
| formatting_failure | 5 | 1.4% |
| llm_timeout | 3 | 0.9% |
| no_predictions_no_gt | 2 | 0.6% |

### Missed vs Irrelevant

- **Total True Positives**: 379
- **Total False Negatives (missed risks)**: 3340
- **Total False Positives (irrelevant/hallucinated)**: 482
- Of all errors, **87.4%** were missed risks and **12.6%** were irrelevant predictions.

### Worst-Performing Policies (by F1)

| Policy ID | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| 10.0 | 0.0 | 0.0 | 0.0 | 0 | 1 | 3 |
| 100.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 29 |
| 101.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 10 |
| 102.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 27 |
| 103.0 | 0.0 | 0.0 | 0.0 | 0 | 1 | 1 |

### Best-Performing Policies (by F1)

| Policy ID | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| 120.0 | 1.0 | 1.0 | 1.0 | 1 | 0 | 0 |
| 128.0 | 1.0 | 1.0 | 1.0 | 1 | 0 | 0 |
| 169.0 | 1.0 | 1.0 | 1.0 | 4 | 0 | 0 |
| 170.0 | 1.0 | 1.0 | 1.0 | 2 | 0 | 0 |
| 191.0 | 1.0 | 1.0 | 1.0 | 2 | 0 | 0 |

## fixed_500

Total policies processed: **349**

| Error Category | Count | % |
|---|---|---|
| missed_risk_only | 167 | 47.9% |
| mixed_errors | 140 | 40.1% |
| irrelevant_retrieval_only | 17 | 4.9% |
| perfect_match | 14 | 4.0% |
| no_predictions_no_gt | 4 | 1.1% |
| formatting_failure | 4 | 1.1% |
| llm_timeout | 3 | 0.9% |

### Missed vs Irrelevant

- **Total True Positives**: 375
- **Total False Negatives (missed risks)**: 3371
- **Total False Positives (irrelevant/hallucinated)**: 446
- Of all errors, **88.3%** were missed risks and **11.7%** were irrelevant predictions.

### Worst-Performing Policies (by F1)

| Policy ID | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| 10.0 | 0.0 | 0.0 | 0.0 | 0 | 12 | 3 |
| 100.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 29 |
| 101.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 10 |
| 102.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 27 |
| 103.0 | 0.0 | 0.0 | 0.0 | 0 | 2 | 1 |

### Best-Performing Policies (by F1)

| Policy ID | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| 120.0 | 1.0 | 1.0 | 1.0 | 1 | 0 | 0 |
| 128.0 | 1.0 | 1.0 | 1.0 | 1 | 0 | 0 |
| 158.0 | 1.0 | 1.0 | 1.0 | 2 | 0 | 0 |
| 170.0 | 1.0 | 1.0 | 1.0 | 2 | 0 | 0 |
| 191.0 | 1.0 | 1.0 | 1.0 | 2 | 0 | 0 |

## semantic

Total policies processed: **349**

| Error Category | Count | % |
|---|---|---|
| mixed_errors | 271 | 77.7% |
| missed_risk_only | 34 | 9.7% |
| irrelevant_retrieval_only | 23 | 6.6% |
| llm_timeout | 9 | 2.6% |
| perfect_match | 6 | 1.7% |
| formatting_failure | 4 | 1.1% |
| no_predictions_no_gt | 2 | 0.6% |

### Missed vs Irrelevant

- **Total True Positives**: 918
- **Total False Negatives (missed risks)**: 2638
- **Total False Positives (irrelevant/hallucinated)**: 934
- Of all errors, **73.9%** were missed risks and **26.1%** were irrelevant predictions.

### Worst-Performing Policies (by F1)

| Policy ID | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| 10.0 | 0.0 | 0.0 | 0.0 | 0 | 2 | 3 |
| 103.0 | 0.0 | 0.0 | 0.0 | 0 | 3 | 1 |
| 11.0 | 0.0 | 0.0 | 0.0 | 0 | 5 | 3 |
| 111.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 18 |
| 126.0 | 0.0 | 0.0 | 0.0 | 0 | 1 | 1 |

### Best-Performing Policies (by F1)

| Policy ID | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| 120.0 | 1.0 | 1.0 | 1.0 | 1 | 0 | 0 |
| 133.0 | 1.0 | 1.0 | 1.0 | 1 | 0 | 0 |
| 265.0 | 1.0 | 1.0 | 1.0 | 1 | 0 | 0 |
| 278.0 | 1.0 | 1.0 | 1.0 | 2 | 0 | 0 |
| 315.0 | 1.0 | 1.0 | 1.0 | 1 | 0 | 0 |