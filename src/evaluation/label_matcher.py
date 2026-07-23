"""
Match LLM-predicted practices against APP-350 ground truth.

Confirmed real schema (checked against the actual v1.1 release files):

    policy_id: 316
    policy_name: SoundHound
    segments:
    - segment_id: 0
      segment_text: "..."
      annotations:
      - practice: Identifier_Cookie_or_similar_Tech_1stParty
        modality: PERFORMED
      sentences: [...]

So ground-truth labels live under segments[*].annotations[*].practice,
not a flat top-level 'practices' list. Practice names follow a
`<Category>_<Detail>_<1stParty|3rdParty>` convention, e.g.
`Location_GPS_1stParty`, `Contact_E_Mail_Address_1stParty`.

We normalise both the real APP-350 labels and the LLM's predicted labels
to a common lowercase/underscore form so they can be compared directly.
Party (1st/3rd) is kept as part of the label since APP-350 treats
first-party and third-party collection of the same data type as
distinct practices.
"""
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
    """Lowercase, collapse repeated underscores/hyphens/whitespace, then
    map known near-miss variants the local model tends to produce
    (observed via scripts/debug_one.py — e.g. it sometimes says
    'Identifier_Advertising' instead of the vocabulary's 'Identifier_Ad_ID',
    or 'Location_Network' instead of 'Location_IP_Address'). This does NOT
    invent credit for genuinely wrong predictions — it only normalises
    spelling/naming variants of the same underlying practice so the score
    reflects concept accuracy rather than exact string match. Document
    this alias table explicitly in your Methodology chapter, since it's
    a scoring decision that affects your reported numbers."""
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
    # LLM commonly writes "Email" instead of APP-350's canonical "E_Mail"
    "contact_email_address": "contact_e_mail_address",
    # LLM sometimes writes "Contact_Mail" or "Contact_Email"
    "contact_email": "contact_e_mail_address",
    "contact_mail": "contact_e_mail_address",
}


def extract_ground_truth_labels(ground_truth_yaml: dict) -> set[str]:
    """Real APP-350 schema: labels are nested under
    segments[*].annotations[*].practice, not a flat top-level list.
    Only PERFORMED modality counts as ground truth for our precision/
    recall comparison (NOT_PERFORMED is a distinct annotated claim about
    the policy explicitly denying a practice, and isn't what our
    extraction prompt is asking the model to find)."""
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
