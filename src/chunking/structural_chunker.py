"""
Structural chunking: split on the "## <heading>" markers inserted by
html_cleaner.py, so each chunk corresponds to one policy section
(e.g. "Information We Collect", "Sharing"). Long sections are further
split by a word-count cap so no single chunk is unreasonably large;
short adjacent sections are left separate (no merging) to keep section
identity traceable for evidence citation.
"""
from __future__ import annotations
import re
from .base import Chunk

HEADING_PATTERN = re.compile(r"^## (.+)$", re.MULTILINE)
MAX_SECTION_WORDS = 400  # sub-split guard for unusually long sections


def chunk_structural(policy_id: str, text: str, config_name: str = "structural") -> list[Chunk]:
    sections = _split_on_headings(text)
    chunks: list[Chunk] = []
    position = 0

    for heading, body in sections:
        words = body.split()
        if len(words) <= MAX_SECTION_WORDS:
            sub_bodies = [body]
        else:
            sub_bodies = [
                " ".join(words[i : i + MAX_SECTION_WORDS])
                for i in range(0, len(words), MAX_SECTION_WORDS)
            ]

        for sub_body in sub_bodies:
            if not sub_body.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{policy_id}_{config_name}_{position}",
                    policy_id=policy_id,
                    text=sub_body,
                    position=position,
                    config_name=config_name,
                    section_heading=heading,
                    metadata={"strategy": "structural"},
                )
            )
            position += 1

    return chunks


def _split_on_headings(text: str) -> list[tuple[str | None, str]]:
    matches = list(HEADING_PATTERN.finditer(text))
    if not matches:
        return [(None, text)]

    sections: list[tuple[str | None, str]] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append((None, preamble))

    for idx, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((heading, body))

    return sections
