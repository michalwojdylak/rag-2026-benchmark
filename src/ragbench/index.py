"""Build the shared OpenSearch indexes for one dataset (passages, sentences, chunks)."""

import time

import numpy as np

from ragbench import config, data, search
from ragbench.embed import embed_docs


def _embeddings(ds, kind, texts):
    path = config.DATA_PROCESSED / ds / f"{kind}_emb.npy"
    if path.exists():
        emb = np.load(path)
        if len(emb) == len(texts):
            return emb
    path.parent.mkdir(parents=True, exist_ok=True)
    emb = embed_docs(texts).astype(np.float32)
    np.save(path, emb)
    return emb


def build_base(ds, recreate=False):
    search.ensure_pipeline()
    stats = {}
    for kind, items in (("passages", data.passages(ds)), ("sentences", data.sentences(ds))):
        name = f"{ds}-{kind}"
        if not recreate and search.count(name) == len(items):
            continue
        t0 = time.time()
        emb = _embeddings(ds, kind, [x["text"] for x in items])
        t_emb = time.time() - t0
        search.create_index(name)
        search.bulk_index(name, ({**x, "embedding": e.tolist()} for x, e in zip(items, emb)))
        stats[kind] = {"docs": len(items), "embed_seconds": round(t_emb, 1), "total_seconds": round(time.time() - t0, 1)}
        print(f"[{ds}] {kind}: {stats[kind]}")
    name = f"{ds}-chunks"
    ch = data.chunks(ds)
    if recreate or search.count(name) != len(ch):
        search.create_index(name, vector=False)
        search.bulk_index(name, ({"id": cid, "chunk_id": cid, "text": t} for cid, t in ch.items()))
    return stats
