from __future__ import annotations
from .base import Chunk
def chunk_fixed(
    policy_id: str,
    text: str,
    chunk_size_words: int,
    overlap_words: int,
    config_name: str,
) -> list[Chunk]:
    words = text.split()
    chunks: list[Chunk] = []
    step = max(chunk_size_words - overlap_words, 1)
    position = 0
    for i in range(0, len(words), step):
        window = words[i : i + chunk_size_words]
        if not window:
            continue
        chunk_text = " ".join(window)
        chunks.append(
            Chunk(
                chunk_id=f"{policy_id}_{config_name}_{position}",
                policy_id=policy_id,
                text=chunk_text,
                position=position,
                config_name=config_name,
                metadata={"chunk_size_words": chunk_size_words, "overlap_words": overlap_words},
            )
        )
        position += 1
        if i + chunk_size_words >= len(words):
            break
    return chunks