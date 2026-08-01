from __future__ import annotations
import re
import numpy as np
from .base import Chunk
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
def chunk_semantic(
    policy_id: str,
    text: str,
    embed_fn,
    similarity_threshold: float = 0.55,
    min_chunk_words: int = 40,
    max_chunk_words: int = 250,
    config_name: str = "semantic",
) -> list[Chunk]:
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [
            Chunk(
                chunk_id=f"{policy_id}_{config_name}_0",
                policy_id=policy_id,
                text=sentences[0],
                position=0,
                config_name=config_name,
            )
        ]
    embeddings = embed_fn(sentences)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    chunks: list[Chunk] = []
    current_sentences = [sentences[0]]
    current_words = len(sentences[0].split())
    position = 0
    for i in range(1, len(sentences)):
        sim = float(np.dot(embeddings[i - 1], embeddings[i]))
        sent_words = len(sentences[i].split())
        breakpoint_hit = sim < similarity_threshold and current_words >= min_chunk_words
        size_cap_hit = current_words + sent_words > max_chunk_words
        if breakpoint_hit or size_cap_hit:
            chunks.append(_make_chunk(policy_id, current_sentences, position, config_name))
            position += 1
            current_sentences = [sentences[i]]
            current_words = sent_words
        else:
            current_sentences.append(sentences[i])
            current_words += sent_words
    if current_sentences:
        chunks.append(_make_chunk(policy_id, current_sentences, position, config_name))
    return chunks
def _make_chunk(policy_id: str, sentences: list[str], position: int, config_name: str) -> Chunk:
    return Chunk(
        chunk_id=f"{policy_id}_{config_name}_{position}",
        policy_id=policy_id,
        text=" ".join(sentences),
        position=position,
        config_name=config_name,
        metadata={"n_sentences": len(sentences)},
    )