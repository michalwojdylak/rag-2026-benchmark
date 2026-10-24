# Agentic RAG, Graph-R1, MetaRAG and friends on one search engine: what actually helps in 2026?

*Draft for the blog post. Every number comes from the per-question JSONL files in `results/`.*

## TL;DR

* **Turning on the generator's reasoning moved accuracy more than any retrieval architecture.** On the same 200
  multi-hop questions, `gpt-5.4-mini` with `reasoning_effort="medium"` lifted naive RAG from 56 % to 78 % and the
  agentic loops from ~66 % to **85–86 %**. Reasoning also made the agents *search more* (A-RAG: 2.1 → 6.0 searches
  per question).
* **With reasoning off, nine quite different systems land within a few points of each other.** On the multi-hop
  sets MetaRAG (72.5 %), LightRAG (71.3 %) and plain hybrid search + a cross-encoder (71.2 %) share the top, with
  the agents (68 %) and one-shot hybrid search (67 %) right behind — differences mostly inside the confidence intervals.
* **The cheapest strong system is boring:** OpenSearch hybrid query (BM25 + k-NN, RRF) → local `bge-reranker-v2-m3`
  → one LLM call. $1.26 per 1,000 questions. LightRAG matches it at 14× the per-query cost plus a ~$20 graph build.
* **Graph-R1 is only as good as its RL-trained policy**, which is not released. With a prompted policy its
  think/query loop asks one question and stops (93 % of 2Wiki questions), landing at 43 % on multi-hop vs 67 % for naive RAG.
* **MetaRAG is a reasoning prosthesis:** best on 2Wiki with reasoning off (79 %), nearly useless once the model reasons.
* **An RL-trained 7B (Search-R1) on a MacBook** reaches 64 % on multi-hop, 4 pp behind `gpt-5.4-mini` naive RAG on
  the same questions (tied on MuSiQue), at $0 API cost and 1–2 minutes per question.
* **Check your benchmark first:** on GraphRAG-Bench Medical the LLM alone scores 90 %.

## Why another comparison

RAG papers each bring their own retriever, corpus split, generator and metric, so their tables don't compose.
The practical question is narrower: *if my search engine is fixed, which way of asking it questions works best?*
Everything except the retrieval strategy is held constant:

* one **OpenSearch 2.19** node — BM25, Lucene HNSW vectors, the native `hybrid` query with an RRF search pipeline,
  phrase search with highlighting;
* one embedding model (`bge-base-en-v1.5`, local), one generator (`gpt-5.4-mini`), one judge (`gpt-4.1-mini`, a
  different model family, using A-RAG's correct/incorrect rubric);
* the unified datasets released with A-RAG (MuSiQue, HotpotQA, 2WikiMultiHopQA from the LinearRAG setup, plus
  GraphRAG-Bench Medical and Novel), each with its own closed corpus, 200 sampled questions each.

Where a paper's contribution is *how to ask for data* — agentic RAG, A-RAG, MetaRAG, Search-R1 — its tools are
backed by the same OpenSearch indexes. Only the graph methods build a second index, because that *is* their idea.

## The contenders

| Method | Idea | Source |
|---|---|---|
| closed book | the LLM alone | – |
| naive RAG | one hybrid query, top-8 passages | OpenSearch |
| rerank RAG | hybrid top-30 → cross-encoder → top-8 | + `bge-reranker-v2-m3` |
| MetaRAG | answer, *monitor* against an independent expert, *critic* classifies the error, *planner* rewrites the query | Zhou et al., WWW'24 (via the SIGIR'26 reproduction) |
| agentic RAG | the LLM calls one `search` tool in a loop | the pattern from the 2025 agentic RAG survey |
| A-RAG | three tools at different granularity: exact keywords, sentence vectors, full chunk reading | arXiv 2602.03442 (Feb 2026) |
| Search-R1 | 7B model trained with PPO to interleave `<think>`, `<search>`, `<answer>` | Jin et al., 2025 |
| LightRAG | LLM-extracted entity/relation graph, local + global keyword retrieval | HKUDS, `lightrag-hku` |
| Graph-R1 | n-ary knowledge hypergraph + a think → query → knowledge → answer agent | Luo et al., ICML'26 |

AgenticRAG-R1, which I wanted to include, turned out to be a GRPO training framework without a released checkpoint;
Search-R1 represents that family because its weights are public.

## Results (reasoning off)

![LLM-Acc heatmap](../results/figures/accuracy_heatmap.png)

| | MuSiQue | HotpotQA | 2Wiki | Medical | Novel | multi-hop avg | $/1k q |
|---|---|---|---|---|---|---|---|
| closed book | 14.0 | 48.5 | 43.5 | 90.0 | 43.0 | 35.3 | 0.17 |
| naive RAG | 49.5 | 86.5 | 65.5 | 88.5 | 72.0 | 67.2 | 1.25 |
| rerank RAG | **58.0** | **87.0** | 68.5 | 89.0 | **78.5** | 71.2 | 1.26 |
| MetaRAG | 54.5 | 84.0 | **79.0** | 84.5 | 68.0 | **72.5** | 3.14 |
| agentic RAG | 54.5 | 85.5 | 64.0 | **92.0** | 73.5 | 68.0 | 2.71 |
| A-RAG | 51.5 | 81.5 | 71.5 | 91.5 | 74.0 | 68.2 | 6.68 |
| LightRAG | 57.5 | 86.0 | 70.5 | 91.0 | 76.0 | 71.3 | 17.41 + build |
| Graph-R1 (prompted) | 26.0 | 60.0 | 42.0 | 85.0 | 48.5 | 42.7 | 0.84 + build |
| Search-R1 7B (n=100) | 47.0 | 84.0 | 61.0 | – | – | 64.0 | 0 (local) |

95 % bootstrap CIs are roughly ±6–7 pp per cell (full table with CIs in the README).

![Accuracy vs cost](../results/figures/accuracy_vs_cost.png)

## Finding 1 — the generator's reasoning is the biggest lever

![Reasoning on vs off](../results/figures/reasoning_effect.png)

The main runs use `reasoning_effort="none"` for every method: `gpt-5.4-mini` refuses function tools combined with
reasoning on Chat Completions, and one setting keeps the methods comparable. Experiment 2 re-runs six methods on the
first 100 MuSiQue and 100 2Wiki questions with `"medium"` through the Responses API.

| method | reasoning off | medium | searches / q (off → medium) | $/1k q (off → medium) |
|---|---|---|---|---|
| closed book | 30.0 | 54.0 | – | 0.10 → 8.57 |
| naive RAG | 56.0 | 78.0 | 1 → 1 | 1.17 → 3.39 |
| MetaRAG | 68.5 | 74.0 | 1.3 → 1.2 | 3.86 → 7.99 |
| agentic RAG | 65.5 | 85.0 | 2.4 → 4.1 | 3.32 → 10.07 |
| A-RAG | 66.0 | **86.5** | 2.1 → 6.0 | 6.96 → 19.65 |
| Graph-R1 | 33.5 | 62.0 | 1.8 → 2.2 | 0.85 → 7.25 |

Two things change. The agents stop making *answer-type* mistakes — with reasoning off, asked "Which film has the
director who died earlier, X or Y?", they often search correctly and then answer with the director's name. And they
explore more: A-RAG triples its searches. A simple one-tool agent loop goes from 70 % to 94 % on 2Wiki.

Reasoning is not free: A-RAG goes from $7 to $20 per 1,000 questions. Naive RAG with reasoning ($3.4) is the
cost-efficient point if you can live with 78 %.

## Finding 2 — MetaRAG is a reasoning prosthesis

MetaRAG checks its own answers: an independent "expert" answers too; if the two disagree, a critic classifies the
error ("the question asks *whose* residence, not which building") and a planner rewrites the query. With reasoning
off this is the best method on 2Wiki (79 % vs 65.5 % naive), at 4–6 LLM calls per question. With reasoning on it
adds almost nothing (68.5 → 74 %) and ends below naive RAG with reasoning (78 %): the model now does that checking itself.

## Finding 3 — graphs: LightRAG is solid but heavy, Graph-R1 needs its policy

**LightRAG** is in the top cluster (71.3 % multi-hop) and has the best context recall of any method (82.5 % of
short answers appear in its context), but its default context is ~23k tokens per question: $17.4 per 1,000 questions,
14× naive RAG, plus ~$20 of `gpt-4o-mini` to build graphs for the five corpora. A reranker gets the same accuracy for
the price of naive RAG.

**Graph-R1** builds a hypergraph of "knowledge segments" (n-ary facts as sentences, $8.4 for all five corpora) and
lets an agent query it. Its authors train that agent with GRPO; without the weights the policy here is the prompted
generator, and it collapses: one query for 93 % of 2Wiki questions, 42.7 % on multi-hop. An ablation gives the same
hypergraph to the function-calling loop of `agentic_rag`: 3× the queries and +17 pp (59.7 %), still below plain
passage retrieval, because rewritten facts drop details that raw passages keep (context recall 59 % vs 65–83 %).
Reasoning helps it most of all (+28 pp), but even then it issues about one query. Graph-R1's index is only as good
as the policy that explores it, which is exactly what its own RL ablation says.

## Finding 4 — an RL-trained 7B on a laptop

Search-R1 (Qwen2.5-7B trained with PPO on NQ + HotpotQA; Q4_K_M GGUF on llama.cpp / Metal) uses the same
OpenSearch hybrid retriever through its `<search>` tags, top-3 passages per search, greedy decoding.

| | MuSiQue | HotpotQA | 2Wiki | multi-hop |
|---|---|---|---|---|
| Search-R1 7B | 47 | 84 | 61 | 64.0 |
| naive RAG, `gpt-5.4-mini`, same 100 questions | 47 | 91 | 65 | 67.7 |

It searches ~3 times per question (2.6–3.2) and its contexts contain the answer 73.7 % of the time, the same as the
frontier-model `agentic_rag` loop (73.5 %): **the RL-trained policy decides when and what to search as well as a much
larger prompted model; it loses on reading.** HotpotQA, which it was trained on, is its best dataset. It is the only
method with zero API cost, and the slowest: 70–100 s median per question on an M3 that was also running other jobs.

## Finding 5 — benchmark hygiene

* **Medical (GraphRAG-Bench) is saturated:** closed book 90 %, every method 84–92 %. It measures medical knowledge.
* **MuSiQue is the discriminating dataset:** closed book 14 %, best non-reasoning method 58 %, best reasoning one 81 %.
* The LLM judge is lenient on near misses ("early 1980s" accepted for "1981"); EM / F1 / Contain are reported next
  to it for the short-answer sets; they reorder methods inside the top cluster but agree on who is in it
  (reranked hybrid, MetaRAG, LightRAG) and on the bottom (closed book, prompted Graph-R1).

## Engineering notes (things that cost hours)

* `gpt-5.4-mini` + function tools + `reasoning_effort` other than `"none"` is rejected on `/v1/chat/completions`.
  Use `/v1/responses` and replay the encrypted reasoning items between tool turns
  (`include=["reasoning.encrypted_content"]`, `store=False`).
* `Qwen3-Embedding-0.6B` (A-RAG's choice) embedded ~5 passages/s on an M3 via MPS; `bge-base-en-v1.5` ~50/s.
* LightRAG with its default file storages commits every store after every document, so indexing slows quadratically.
  Insert many corpus chunks per LightRAG document, micro-batch its embedding calls, and check for FAILED documents
  afterwards: they are not retried, and an interrupted build can leave one that cannot be repaired.
* OpenSearch's `hybrid` query + `score-ranker-processor` (RRF) pipeline is a few lines of JSON and was the most
  valuable single component in the study.

## Limitations

* One generator family and one run per configuration; at n = 200, differences under ~5 pp are noise.
* Graph-R1 and Agentic-RAG-R1 are evaluated without their RL-trained policies (none released).
* Search-R1 runs a 4-bit checkpoint on 100 questions per multi-hop dataset, with our retriever instead of the
  E5 + Wikipedia setup it was trained against, and is not evaluated on the long-answer sets.
* MetaRAG's T5 helper models are replaced by `gpt-5.4-nano`; LightRAG's MuSiQue graph misses 4.7 % of the corpus.

## Reproduce

See the [README](../README.md#reproduce). Total API spend for everything here: ≈ $80.
