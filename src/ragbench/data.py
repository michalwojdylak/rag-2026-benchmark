"""Load the A-RAG / LinearRAG / GraphRAG-Bench corpora and questions in one format.

Corpus layout (identical for every method, so only the method differs):
  chunk    - the original corpus unit from `chunks.json` (several paragraphs, ~1k tokens)
  passage  - a ~150-word window of whole sentences inside one chunk; the unit most retrievers index
  sentence - single sentences, used by A-RAG's semantic search
"""

import json
import random
import re
from collections import defaultdict
from functools import lru_cache

from ragbench import config

GRAPHRAG_BENCH_TYPES = ("Fact Retrieval", "Complex Reasoning")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@lru_cache(maxsize=None)
def chunks(ds):
    raw = json.load(open(config.DATA_RAW / ds / "chunks.json"))
    out = {}
    for item in raw:
        cid, text = item.split(":", 1)
        out[cid] = text.strip()
    return out


def split_sentences(text):
    return [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 10]


@lru_cache(maxsize=None)
def passages(ds):
    """[{id, chunk_id, text}] with windows of whole sentences up to PASSAGE_WORDS words."""
    out = []
    for cid, text in chunks(ds).items():
        buf, n = [], 0
        for s in split_sentences(text):
            w = len(s.split())
            if buf and n + w > config.PASSAGE_WORDS:
                out.append({"id": f"{cid}-{len(out)}", "chunk_id": cid, "text": " ".join(buf)})
                buf, n = [], 0
            buf.append(s)
            n += w
        if buf:
            out.append({"id": f"{cid}-{len(out)}", "chunk_id": cid, "text": " ".join(buf)})
    return out


@lru_cache(maxsize=None)
def sentences(ds):
    return [{"id": f"{cid}-s{i}", "chunk_id": cid, "text": s}
            for cid, text in chunks(ds).items() for i, s in enumerate(split_sentences(text))]


def questions(ds, n=None, seed=42):
    """Deterministic sample; GraphRAG-Bench sets are stratified over GRAPHRAG_BENCH_TYPES."""
    qs = json.load(open(config.DATA_RAW / ds / "questions.json"))
    if n is None or n >= len(qs):
        return qs
    rng = random.Random(seed)
    if ds in config.SHORT_ANSWER:
        return rng.sample(qs, n)
    by_type = defaultdict(list)
    for q in qs:
        # Summaries and creative writing have no single correct answer to judge against.
        if q["question_type"] in GRAPHRAG_BENCH_TYPES:
            by_type[q["question_type"]].append(q)
    types = sorted(by_type)
    picked = []
    for i, t in enumerate(types):
        k = n // len(types) + (1 if i < n % len(types) else 0)
        picked += rng.sample(by_type[t], min(k, len(by_type[t])))
    return picked

