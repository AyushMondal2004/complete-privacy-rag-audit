"""
Thin wrapper around sentence-transformers so the rest of the codebase
doesn't depend directly on the library (makes it easy to swap models
for an ablation if you want one).
"""
from __future__ import annotations
import os
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    model = _get_model(model_name)
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
