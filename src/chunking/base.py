from dataclasses import dataclass, field
@dataclass
class Chunk:
    chunk_id: str
    policy_id: str
    text: str
    position: int
    config_name: str
    section_heading: str | None = None
    metadata: dict = field(default_factory=dict)