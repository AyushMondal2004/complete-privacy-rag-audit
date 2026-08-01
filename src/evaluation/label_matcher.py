from __future__ import annotations
import re
from dataclasses import dataclass
@dataclass
class MatchResult:
    policy_id: str
    true_positives: list[str]
    false_positives: list[str]
    false_negatives: list[str]
def normalise_label(label: str) -> str:
    label = label.strip().lower()
    label = re.sub(r"[\s\-]+", "_", label)
    label = re.sub(r"_+", "_", label)
    for pattern, replacement in LABEL_ALIASES.items():
        if pattern in label:
            party = "_3rdparty" if "3rdparty" in label else "_1stparty"
            return replacement + party
    return label
LABEL_ALIASES = {
    "identifier_advertising": "identifier_ad_id",
    "identifier_ad": "identifier_ad_id",
    "location_network": "location_ip_address",
    "location_wifi_network": "location_wifi",
    "identifier_cookie": "identifier_cookie_or_similar_tech",
    "contact_email_address": "contact_e_mail_address",
    "contact_email": "contact_e_mail_address",
    "contact_mail": "contact_e_mail_address",
}
def extract_ground_truth_labels(ground_truth_yaml: dict) -> set[str]:
    labels = set()
    for segment in ground_truth_yaml.get("segments", []):
        for annotation in segment.get("annotations", []):
            practice = annotation.get("practice")
            modality = annotation.get("modality", "PERFORMED")
            if practice and modality == "PERFORMED":
                labels.add(normalise_label(practice))
    return labels
def extract_predicted_labels(llm_json: dict) -> set[str]:
    labels = set()
    for p in llm_json.get("practices", []):
        if p.get("modality") == "PERFORMED":
            practice = p.get("practice", "")
            if practice:
                labels.add(normalise_label(practice))
    return labels
def match_labels(policy_id: str, predicted: set[str], ground_truth: set[str]) -> MatchResult:
    tp = sorted(predicted & ground_truth)
    fp = sorted(predicted - ground_truth)
    fn = sorted(ground_truth - predicted)
    return MatchResult(policy_id=policy_id, true_positives=tp, false_positives=fp, false_negatives=fn)