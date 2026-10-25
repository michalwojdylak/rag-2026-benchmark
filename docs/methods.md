# Methods: what each idea is and how it is implemented here

Every method answers the same questions over the same corpus, with the same generator
(`gpt-5.4-mini`, `reasoning_effort="none"`), the same embedding model (`BAAI/bge-base-en-v1.5`)
and the same search engine (OpenSearch 2.19). Only the *retrieval strategy* differs.
Where a paper's idea is "how to ask for data" rather than a new index, the index is plain OpenSearch.

| Method | Family | Paper / code | What is faithful | What is adapted |
|---|---|---|---|---|
| `closed_book` | baseline | - | - | - |
| `naive_rag` | baseline | - | OpenSearch native hybrid query (BM25 + k-NN, RRF pipeline), top-8 passages | - |
| `rerank_rag` | baseline | bge-reranker-v2-m3 | hybrid top-30 → cross-encoder → top-8 | - |
| `metarag` | metacognitive | Zhou et al., WWW'24 ([arXiv 2402.11626](https://arxiv.org/abs/2402.11626)); code via the SIGIR'26 reproducibility study ([iai-group/sigir2026-metarag](https://github.com/iai-group/sigir2026-metarag)) | monitor → evaluate → plan loop, all prompts, threshold 0.3, 3 iterations, top-5 references, MiniLM similarity | T5-large "expert" QA model and T5-XXL NLI model replaced by `gpt-5.4-nano` with the same role; retriever is our hybrid search instead of Wikipedia BM25 |
| `agentic_rag` | agentic | "Agentic RAG" pattern ([survey, arXiv 2501.09136](https://arxiv.org/abs/2501.09136)) | one `search` tool (hybrid, top-5), function-calling loop, ≤10 turns | - (it is a pattern, not a system) |
| `a_rag` | agentic | A-RAG, Feb 2026 ([arXiv 2602.03442](https://arxiv.org/abs/2602.03442), [code](https://github.com/Ayanami0730/arag)) | three tools `keyword_search` / `semantic_search` / `read_chunk`, tool descriptions and system prompt verbatim, read-tracking, ≤15 loops, forced final answer | keyword search is OpenSearch `match_phrase` + highlighting (A-RAG: Python substring count); sentence vectors live in an OpenSearch k-NN index (A-RAG: a pickle) |
| `search_r1` | RL-trained | Search-R1 ([arXiv 2503.09516](https://arxiv.org/abs/2503.09516), [code](https://github.com/PeterGriffinJin/Search-R1)) | released PPO checkpoint `SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo`, prompt + `<search>`/`<information>`/`<answer>` protocol from `infer.py`, top-3 docs, greedy decoding | Q4_K_M GGUF quantisation on llama.cpp (Metal); retriever is our hybrid search (paper: E5 over Wikipedia) |
| `lightrag` | graph | LightRAG ([arXiv 2410.05779](https://arxiv.org/abs/2410.05779), `lightrag-hku` 1.5.7) | library indexing pipeline untouched, `hybrid` mode | gpt-4o-mini for extraction; LightRAG's keyword-extraction prompt is called through our tracked client and its context is answered with the shared answer prompt |
| `graph_r1` | graph + agentic | Graph-R1, ICML'26 ([arXiv 2507.21892](https://arxiv.org/abs/2507.21892), [code](https://github.com/LHRLAB/Graph-R1)) | hypergraph extraction prompt (n-ary "knowledge segments" + entities), dual-path retrieval (entity → hyperedges by degree, hyperedge vectors) fused with Σ1/(rank+1), top-10, agent prompt and `<think>/<query>/<knowledge>/<answer>` protocol | no released RL checkpoint → the policy is the prompted generator; one extraction pass instead of 2 gleaning passes; one few-shot example instead of three; hypergraph stored in OpenSearch (paper: FAISS + JSON) |
| `graph_r1_fc` | ablation | - | same hypergraph and retrieval as `graph_r1` | the policy is `agentic_rag`'s function-calling loop (≤10 turns) instead of Graph-R1's tag protocol |

## Experiment 2: reasoning on

`results/runs_reasoning_medium/` re-runs `closed_book`, `naive_rag`, `metarag`, `agentic_rag`, `a_rag` and
`graph_r1` with `RAGBENCH_REASONING=medium` on the first 100 questions of the MuSiQue and 2Wiki samples (the
same questions as the main run, so the comparison is paired). `gpt-5.4-mini` rejects function tools together
with `reasoning_effort` on Chat Completions, so these runs go through the Responses API (`llm._responses`),
replaying the encrypted reasoning items between tool turns. MetaRAG's helper model (`gpt-5.4-nano`) also runs
with medium reasoning in this experiment.

## LightRAG build notes

Default local storages commit every store after each inserted document, which rewrites the whole graph each
time; corpus chunks are therefore inserted 16 (Medical, 2Wiki) or 64 (others) per LightRAG document. LightRAG
re-chunks at 1,200 tokens, so retrieval units are the same either way. Concurrent single-text embedding calls
are micro-batched (`_EmbedBatcher`), and documents left FAILED (vector-store timeouts) are deleted and
re-inserted (`_repair`). One MuSiQue document (corpus chunks 832-895) could not be repaired
(`RecoveryAnchorMissingError` after an interrupted build) and is missing from that graph. LightRAG's
indexing cost is measured on Medical and estimated for the other corpora from LightRAG's own LLM-call cache
(builds were restarted, and cached calls are invisible to our cost tracker).

## Why these and not others

The brief named AgenticRAG-R1, MetaRAG, Agentic RAG and Graph-R1. Findings while choosing:

* **Agentic-RAG-R1** ([jiangxinke/Agentic-RAG-R1](https://github.com/jiangxinke/Agentic-RAG-R1)) is a
  GRPO *training framework* and ships no trained checkpoint, so there is nothing to evaluate without
  multi-GPU RL training. Its idea (an LLM trained with RL to decide when and what to search) is
  covered by **Search-R1**, which uses the same Search-R1 retriever interface and has public weights.
* **Graph-R1** also releases no weights (the HF models called "Graph-R1" belong to an unrelated
  HKUST graph-reasoning paper), so its *retrieval architecture* is evaluated with a prompted policy.
  This isolates "is the hypergraph + agent loop useful?" from "does GRPO training help?".
* **"MetaRAG"** names at least three different things: the WWW'24 metacognitive RAG (used here), a
  2025 metamorphic-testing hallucination detector, and an unrelated PyPI package.
* **A-RAG** (Feb 2026) was added because it is the most recent agentic method with code that reports
  state-of-the-art results on exactly these datasets, so our numbers can be compared with its tables.
* **LightRAG** was added as the most widely used open-source GraphRAG library and as the "graph,
  but not agentic" reference point for Graph-R1.

Considered but not included (candidates for a follow-up): HippoRAG 2 and LinearRAG (strong
structure-based baselines in the A-RAG tables), MemGraphRAG, EHRAG, ConRAG (2026 graph/multi-view
methods without mature code), R1-Searcher / ReSearch / ZeroSearch / s3 (more RL search agents of the
Search-R1 kind), Microsoft GraphRAG (indexing cost far above LightRAG for this corpus size).
