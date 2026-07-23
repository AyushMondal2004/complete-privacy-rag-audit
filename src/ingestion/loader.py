"""
Load the APP-350 corpus from disk.

Confirmed layout (checked against the actual v1.1 release):

    data/raw/APP-350_v1.1/original_documents/<id>.html   (e.g. 316.html)
    data/raw/APP-350_v1.1/annotations/policy_<id>.yml    (e.g. policy_316.yml)

Note the annotation files use a "policy_" prefix and a ".yml" (not
".yaml") extension, and don't match the HTML filename directly — the
loader below strips the prefix and matches on the numeric id.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator
import yaml

RAW_DIR = Path("data/raw")
DOCS_GLOB = "**/original_documents/*.html"
ANNOTATIONS_GLOB = "**/annotations/policy_*.yml"


@dataclass
class RawPolicy:
    policy_id: str
    html_path: Path


def iter_raw_policies(raw_dir: Path = RAW_DIR) -> Iterator[RawPolicy]:
    for html_path in sorted(raw_dir.glob(DOCS_GLOB)):
        policy_id = html_path.stem
        yield RawPolicy(policy_id=policy_id, html_path=html_path)


def load_ground_truth(policy_id: str, raw_dir: Path = RAW_DIR) -> dict | None:
    """Return the parsed YAML annotation for a policy, or None if missing
    (e.g. PDF-only entries like 74.pdf in the corpus have no matching id
    in original_documents as .html — log and skip those)."""
    matches = list(raw_dir.glob(f"**/annotations/policy_{policy_id}.yml"))
    if not matches:
        return None
    with open(matches[0], "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_available_policies(raw_dir: Path = RAW_DIR) -> int:
    return sum(1 for _ in iter_raw_policies(raw_dir))
