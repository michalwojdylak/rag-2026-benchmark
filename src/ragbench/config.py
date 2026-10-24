"""Central settings. Everything a reader may want to change lives here or in env vars."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RUNS_DIR = Path(os.getenv("RAGBENCH_RUNS_DIR", ROOT / "results" / "runs"))
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / ".cache"

DATASETS = ["musique", "hotpotqa", "2wikimultihop", "medical", "novel"]
# Short-answer datasets are scored with EM/F1/Contain; GraphRAG-Bench ones only with the LLM judge.
SHORT_ANSWER = {"musique", "hotpotqa", "2wikimultihop"}

# --- models -----------------------------------------------------------------
GENERATOR = os.getenv("RAGBENCH_GENERATOR", "gpt-5.4-mini")
# "none": gpt-5.4-mini refuses function tools + reasoning on Chat Completions, and one setting keeps methods comparable.
GENERATOR_REASONING = os.getenv("RAGBENCH_REASONING", "none")
JUDGE = os.getenv("RAGBENCH_JUDGE", "gpt-4.1-mini")
# Model used to build graph indexes (LightRAG, Graph-R1 hypergraph); gpt-4o-mini is what those papers used.
INDEXER = os.getenv("RAGBENCH_INDEXER", "gpt-4o-mini")
# Small helper model for MetaRAG's "expert" monitor and NLI checks (originally T5 models).
HELPER = os.getenv("RAGBENCH_HELPER", "gpt-5.4-nano")

# bge-base: Qwen3-Embedding-0.6B (A-RAG's choice) ran at ~5 passages/s on an M3; bge-base does ~50/s.
EMBED_MODEL = os.getenv("RAGBENCH_EMBED", "BAAI/bge-base-en-v1.5")
EMBED_DIM = 768
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = os.getenv("RAGBENCH_RERANK", "BAAI/bge-reranker-v2-m3")

# USD per 1M tokens: (input, cached input, output). Checked 2026-10.
PRICES = {
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
}

# Hard stop for real (non-cached) API spend across all runs. The owner's absolute limit is $100; new calls
# stop at $97 so requests already in flight in parallel processes cannot push the total past it.
ABSOLUTE_LIMIT_USD = 100.0
BUDGET_USD = min(float(os.getenv("RAGBENCH_BUDGET_USD", "97")), ABSOLUTE_LIMIT_USD - 3)

# --- infra --------------------------------------------------------------------
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://localhost:9200")

# --- retrieval defaults shared by all methods --------------------------------
PASSAGE_WORDS = 150
TOP_K = 5
