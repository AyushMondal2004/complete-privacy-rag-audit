from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    policy_id: str
    text: str
    position: int          # order within the policy
    config_name: str       # e.g. "fixed_300", "fixed_500", "semantic"
    section_heading: str | None = None
    metadata: dict = field(default_factory=dict)
