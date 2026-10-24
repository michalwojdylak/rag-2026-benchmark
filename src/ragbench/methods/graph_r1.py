"""Graph-R1 (Luo et al., ICML 2026, arXiv 2507.21892): agentic GraphRAG over a knowledge hypergraph.

Index  - an LLM (gpt-4o-mini, as in the paper) splits every chunk into "knowledge segments"
         (hyperedges: n-ary facts as sentences) and the entities each segment connects
         (Graph-R1's own extraction prompt). Hyperedges and entities live in OpenSearch with vectors.
Query  - dual-path retrieval exactly as graphr1/operate.py: (a) top-5 entities by vector -> their
         hyperedges ranked by degree, (b) top-5 hyperedges by vector; fused with sum of 1/(rank+1),
         top-10 returned as <knowledge>.
Agent  - the think -> <query> -> <knowledge> -> ... -> <answer> loop with Graph-R1's prompt.

Graph-R1 trains a 3B/7B policy with GRPO but has not released checkpoints, so the policy here is
the prompted generator (gpt-5.4-mini). Extraction runs without "gleaning" passes (paper: 2) to cut cost.
"""

import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from ragbench import config, data, llm, search
from ragbench.embed import embed_docs
from ragbench.methods import graph_r1_prompts as P
from ragbench.methods.base import Method, Result, short_trace

AGENT_PROMPT = """Answer the given question. You can query from knowledge base provided to you to answer the question. You can query knowledge as many times as you want.
You must first conduct reasoning inside <think>...</think>. If you need to query knowledge, you can set a query statement between <query>...</query> to query from knowledge base after <think>...</think>.
When you have the final answer, you can output the answer inside <answer>...</answer>.

Output format for tool call:
<think>
...
</think>
<query>
...
</query>

Output format for answer:
<think>
...
</think>
<answer>
...
</answer>
Question: {question}"""

MAX_QUERIES = 6
ENTITY_K = HYPEREDGE_K = 5   # script_api.py
TOP_K = 10                   # QueryParam(top_k=10)


def _clean(s):
    return s.strip().strip('"').strip()


def parse_extraction(text, chunk_id):
    """Graph-R1 record format -> hyperedges with the entities listed after each of them."""
    edges, current = [], None
    for rec in text.replace(P.COMPLETE, "").split(P.RECORD):
        m = re.search(r"\((.*)\)", rec, re.DOTALL)
        if not m:
            continue
        parts = [_clean(x) for x in m.group(1).split(P.TUPLE)]
        if parts[0] == "hyper-relation" and len(parts) >= 2 and parts[1]:
            current = {"text": parts[1], "chunk_id": chunk_id, "entities": [], "entity_desc": {}}
            edges.append(current)
        elif parts[0] == "entity" and len(parts) >= 3 and current is not None:
            name = parts[1].upper()
            if name and name not in current["entities"]:
                current["entities"].append(name)
                current["entity_desc"][name] = (parts[2], parts[3] if len(parts) > 3 else "")
    return edges


class GraphR1(Method):
    name = "graph_r1"
    family = "graph"
    reference = "Graph-R1 (ICML'26) hypergraph + agentic query loop, prompted policy"

    @property
    def idx_edges(self):
        return f"{self.ds}-g1-hyperedges"

    @property
    def idx_entities(self):
        return f"{self.ds}-g1-entities"

    # ------------------------------------------------------------- indexing
    def prepare(self):
        marker = config.DATA_PROCESSED / self.ds / "graph_r1_index.json"
        if marker.exists() and search.count(self.idx_edges) > 0:
            return json.loads(marker.read_text())
        usage = llm.Usage()
        chunks = data.chunks(self.ds)
        prompt_head = P.EXTRACTION.format(language="English", tuple_delimiter=P.TUPLE, record_delimiter=P.RECORD,
                                          completion_delimiter=P.COMPLETE,
                                          examples=P.EXAMPLE.format(tuple_delimiter=P.TUPLE, record_delimiter=P.RECORD,
                                                                    completion_delimiter=P.COMPLETE),
                                          input_text="{input_text}")

        def extract(item):
            cid, text = item
            token = llm.track(usage)
            try:
                out = llm.text(prompt_head.replace("{input_text}", text), model=config.INDEXER)
            finally:
                llm.untrack(token)
            return parse_extraction(out, cid)

        t0 = time.time()
        with ThreadPoolExecutor(32) as ex:
            edges = [e for es in tqdm(ex.map(extract, chunks.items()), total=len(chunks), desc=f"graph-r1 {self.ds}") for e in es]
        t_extract = time.time() - t0

        entities = defaultdict(lambda: {"edges": [], "desc": []})
        for i, e in enumerate(edges):
            e["id"] = f"h{i}"
            for name in e["entities"]:
                entities[name]["edges"].append(e["id"])
                entities[name]["desc"].append(e["entity_desc"][name][1])
        names = list(entities)
        e_emb = embed_docs([e["text"] for e in edges], batch_size=64)
        n_emb = embed_docs(names, batch_size=128)

        search.create_index(self.idx_edges, extra_props={"entities": {"type": "keyword"}})
        search.bulk_index(self.idx_edges, ({"id": e["id"], "text": e["text"], "chunk_id": e["chunk_id"],
                                            "entities": e["entities"], "embedding": v.tolist()}
                                           for e, v in zip(edges, e_emb)))
        search.create_index(self.idx_entities, extra_props={"edges": {"type": "keyword"}, "type": {"type": "keyword"}})
        search.bulk_index(self.idx_entities, ({"id": f"e{i}", "text": n, "edges": entities[n]["edges"],
                                               "embedding": v.tolist()} for i, (n, v) in enumerate(zip(names, n_emb))))
        stats = {"hyperedges": len(edges), "entities": len(names), "extract_seconds": round(t_extract, 1),
                 "index_seconds": round(time.time() - t0, 1), **usage.to_dict()}
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(stats, indent=2))
        return stats

    # ------------------------------------------------------------ retrieval
    def retrieve(self, query):
        ents = search.knn(self.idx_entities, query, k=ENTITY_K)
        edge_ids = list(dict.fromkeys(h for e in ents for h in e["edges"]))
        edge_docs = {d["id"]: d for d in search.get_docs(self.idx_edges, edge_ids)}
        # Graph-R1 sorts the bipartite edges (entity, hyperedge) of the matched entities by edge degree =
        # degree(entity) + degree(hyperedge), highest first.
        rank = {}
        for e in ents:
            for h in e["edges"]:
                if h in edge_docs:
                    rank[h] = max(rank.get(h, 0), len(e["edges"]) + len(edge_docs[h].get("entities", [])))
        path_a = [edge_docs[h] for h in sorted(rank, key=lambda h: -rank[h])]
        path_b = search.knn(self.idx_edges, query, k=HYPEREDGE_K)
        score = defaultdict(float)
        texts = {}
        for path in (path_a, path_b):
            for i, d in enumerate(path):
                score[d["text"]] += 1 / (i + 1)
                texts[d["text"]] = d
        top = sorted(score.items(), key=lambda x: -x[1])[:TOP_K]
        return [{"<knowledge>": t, "<coherence>": round(s, 3)} for t, s in top]

    # ---------------------------------------------------------------- agent
    def answer(self, q):
        messages = [{"role": "user", "content": AGENT_PROMPT.format(question=q["question"]) + "\n\n" + self.style +
                     "\nOutput exactly one <query> or one <answer> per turn and then stop."}]
        contexts, trace = [], []
        for _ in range(MAX_QUERIES + 1):
            out = llm.chat(messages)["content"]
            ans = re.search(r"<answer>(.*?)(?:</answer>|$)", out, re.DOTALL)
            qm = re.search(r"<query>(.*?)(?:</query>|$)", out, re.DOTALL)
            if ans and (not qm or out.find("<answer>") < out.find("<query>")):
                return Result(ans.group(1).strip(), contexts, trace)
            if not qm or len(trace) >= MAX_QUERIES:
                break
            query = qm.group(1).strip()
            try:  # the paper's policy emits {"query": "..."}
                query = json.loads(query).get("query", query)
            except (json.JSONDecodeError, AttributeError):
                pass
            knowledge = self.retrieve(query)
            contexts += [k["<knowledge>"] for k in knowledge]
            kj = json.dumps({"results": knowledge}, ensure_ascii=False)
            trace.append({"query": query, "result": short_trace(kj)})
            messages += [{"role": "assistant", "content": out[: out.find("</query>") + 8] if "</query>" in out else out},
                         {"role": "user", "content": f"<knowledge>\n{kj}\n</knowledge>"}]
        messages.append({"role": "user", "content": "Now give your final answer inside <answer>...</answer>. " + self.style})
        out = llm.chat(messages)["content"]
        ans = re.search(r"<answer>(.*?)(?:</answer>|$)", out, re.DOTALL)
        trace.append({"forced_answer": True})
        return Result((ans.group(1) if ans else out).strip(), contexts, trace)


class GraphR1Tools(GraphR1):
    """Ablation: the same hypergraph retriever, but driven by the function-calling loop of `agentic_rag`
    instead of Graph-R1's tag protocol. Separates "is the hypergraph a good index?" from "does the
    prompted (non-RL) think/query loop explore enough?"."""
    name = "graph_r1_fc"
    reference = "Ablation: Graph-R1 hypergraph retrieval as a function-calling search tool"

    def answer(self, q):
        from ragbench.methods import agent
        from ragbench.methods.agentic import AGENTIC_SYSTEM, _tool

        def search_tool(query):
            knowledge = self.retrieve(query)
            return json.dumps({"results": knowledge}, ensure_ascii=False), [k["<knowledge>"] for k in knowledge]

        tools = {"search": (_tool("search", "Query the knowledge hypergraph. Returns the top knowledge segments for the query.",
                                  {"query": {"type": "string", "description": "Search query"}}, ["query"]), search_tool)}
        ans, ctx, trace = agent.run(AGENTIC_SYSTEM.format(style=self.style), q["question"], tools, 10, self.style)
        return Result(ans, ctx, trace)
