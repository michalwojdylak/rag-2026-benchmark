"""Local open-source embeddings (BGE) and cross-encoder reranker."""

import threading
from functools import lru_cache

import torch

from ragbench import config

_lock = threading.Lock()
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL, device=DEVICE)


def embed_docs(texts, batch_size=32, show_progress=True):
    with _lock:
        return _model().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                               show_progress_bar=show_progress, convert_to_numpy=True)


@lru_cache(maxsize=20000)
def _embed_query(q):
    with _lock:
        return tuple(_model().encode([config.EMBED_QUERY_PREFIX + q], normalize_embeddings=True)[0].tolist())


def embed_query(q):
    return list(_embed_query(q))


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.RERANK_MODEL, device=DEVICE, max_length=512)


def rerank(query, texts):
    """Cross-encoder scores, same order as `texts`."""
    with _lock:
        return _reranker().predict([(query, t) for t in texts], batch_size=16).tolist()
