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
    matches = list(raw_dir.glob(f"**/annotations/policy_{policy_id}.yml"))
    if not matches:
        return None
    with open(matches[0], "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
def count_available_policies(raw_dir: Path = RAW_DIR) -> int:
    return sum(1 for _ in iter_raw_policies(raw_dir))