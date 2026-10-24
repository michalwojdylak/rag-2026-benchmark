"""Tool-calling agents over the shared OpenSearch indexes.

agentic_rag  the generic "Agentic RAG" pattern (Singh et al., 2025 survey): one search tool, ReAct-style loop.
a_rag        A-RAG (Du et al., arXiv 2602.03442, Feb 2026): hierarchical retrieval interfaces -
             keyword_search (exact phrases) + semantic_search (sentence vectors) + read_chunk (full text).
             Tool descriptions and system prompt are copied from github.com/Ayanami0730/arag.
"""

from collections import OrderedDict

from ragbench import data, search
from ragbench.methods import agent
from ragbench.methods.base import Method, Result, format_passages


def _tool(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}}


AGENTIC_SYSTEM = """You are a research assistant that answers questions using a search engine over a private document collection.

Search as many times as you need: break multi-hop questions into sub-questions, search for each, and use what you learn to write the next query. Only answer from the documents you found.

When you are done, reply without calling a tool. {style}"""


class AgenticRAG(Method):
    name = "agentic_rag"
    family = "agentic"
    reference = "ReAct-style tool-calling agent with one hybrid search tool"
    max_loops = 10
    k = 5

    def answer(self, q):
        def search_tool(query):
            hits = search.hybrid(f"{self.ds}-passages", query, k=self.k)
            return format_passages(hits), [h["text"] for h in hits]

        tools = {"search": (_tool("search", "Search the document collection. Returns the top passages for the query.",
                                  {"query": {"type": "string", "description": "Search query"}}, ["query"]), search_tool)}
        ans, ctx, trace = agent.run(AGENTIC_SYSTEM.format(style=self.style), q["question"], tools, self.max_loops, self.style)
        return Result(ans, ctx, trace)


# ---------------------------------------------------------------- A-RAG ----
A_RAG_SYSTEM = """You are a question-answering assistant with access to a document corpus through available tools.

Your goal is to answer questions accurately by finding and analyzing relevant information from the provided documents.

## Available Tools

- **keyword_search**: Find chunks by exact keyword matching
- **semantic_search**: Find chunks by semantic similarity
- **read_chunk**: Read the full content of a specific chunk

## Strategy

Work iteratively: search → read → evaluate → search → read → ... → answer. For multi-hop questions, decompose the problem and tackle each sub-question step by step.

## When Answering

- Ground your response in the retrieved documents
- Provide clear, direct answers supported by evidence
- Avoid speculation beyond what the documents support
- {style}"""

KEYWORD_DESC = """Search for document chunks using keyword-based exact text matching (case-insensitive). Returns chunk IDs and abbreviated sentence snippets where the keywords appear.

IMPORTANT: This tool matches keywords literally in the text. Use SHORT, SPECIFIC terms (1-3 words maximum). Each keyword is matched independently.

Examples of GOOD keywords:
  - Entity names: "Albert Einstein", "Tesla", "Python", "Argentina"
  - Technical terms: "photosynthesis", "quantum mechanics"
  - Key concepts: "climate change", "GDP growth"

Examples of BAD keywords (DO NOT use):
  - Long phrases: "the person who invented the telephone" → use "Alexander Bell" instead
  - Questions: "when did World War 2 start" → use "World War 2", "1939" instead
  - Descriptions: "the country between France and Spain" → use "Andorra" instead
  - Full sentences: "how does the stock market work" → use "stock market", "trading" instead

RETURNS: Abbreviated snippets marked with "..." showing where keywords appear. These snippets help you identify relevant chunks, but you MUST use read_chunk to get the full text for answering questions."""

SEMANTIC_DESC = """Semantic search using embedding similarity. Matches your query against sentences in each chunk via vector similarity.

WHEN TO USE:
- When keyword search fails to find relevant information
- When exact wording in documents is unknown
- For conceptual/meaning-based matching

RETURNS: Abbreviated snippets with matched sentences. Use read_chunk to get full text for answering."""

READ_DESC = """Read the complete content of document chunks by their IDs.

This tool returns the full text of the specified chunks, allowing you to examine the complete context and details that are not visible in search snippets.

IMPORTANT: Search results (keyword_search and semantic_search) only show abbreviated snippets marked with "..." - they are NOT sufficient for answering questions. You MUST use read_chunk to get the full content before formulating your answer.

STRATEGY:
- Always read promising chunks identified by your searches
- Make sure to read the most relevant chunks to gather complete information
- If information seems incomplete or truncated, read adjacent chunks (± 1)
- Reading full text is essential for accurate answers

Note: Previously read chunks will be marked as already seen to avoid redundant information."""

TOP_K_PROP = {"type": "integer", "description": "Number of top-ranked chunks to return (default: 5, max: 20)", "default": 5}


class ARAG(Method):
    name = "a_rag"
    family = "agentic"
    reference = "A-RAG (arXiv 2602.03442): keyword / semantic / read_chunk tools"
    max_loops = 15  # paper default

    def answer(self, q):
        read = set()
        chunks = data.chunks(self.ds)

        def keyword_search(keywords, top_k=5):
            hits = search.phrase_search(f"{self.ds}-chunks", keywords, k=min(int(top_k), 20))
            if not hits:
                return f"No results found for keywords: {keywords}", []
            parts = [f"Chunk ID: {h['id']}, Matched keywords in chunk: ... " + " ... ".join(h.get("highlight", {}).get("text", [])) + " ..."
                     for h in hits]
            return "\n\n".join(parts), parts

        def semantic_search(query, top_k=5):
            top_k = min(int(top_k), 20)
            hits = search.knn(f"{self.ds}-sentences", query, k=top_k * 3)
            grouped = OrderedDict()
            for h in hits:  # hits arrive best-first, so the first sentence per chunk is its max similarity
                grouped.setdefault(h["chunk_id"], []).append(h)
            parts = []
            for cid, sents in list(grouped.items())[:top_k]:
                matched = " ... ".join(s["text"] for s in sents)
                parts.append(f"Chunk ID: {cid} (Similarity: {sents[0]['score']:.3f})\nMatched: ... {matched} ...")
            return ("\n\n".join(parts) or f"No results for: {query}"), parts

        def read_chunk(chunk_ids=None, chunk_id=None):
            ids = [str(c) for c in (chunk_ids or ([chunk_id] if chunk_id is not None else []))]
            parts, ctx = [], []
            for cid in ids:
                if cid in read:
                    parts.append(f"[Chunk {cid}]\n(This chunk has been read before)")
                elif cid in chunks:
                    read.add(cid)
                    parts.append(f"[Chunk {cid}]\n{chunks[cid]}")
                    ctx.append(chunks[cid])
                else:
                    parts.append(f"[Chunk {cid}] - Not found")
            return "\n\n".join(parts) or "Error: No chunk IDs provided", ctx

        tools = {
            "keyword_search": (_tool("keyword_search", KEYWORD_DESC, {
                "keywords": {"type": "array", "items": {"type": "string"},
                             "description": "List of keywords to search. Each keyword should be 1-3 words maximum (e.g., ['Einstein', 'relativity theory', '1905'])."},
                "top_k": TOP_K_PROP}, ["keywords"]), keyword_search),
            "semantic_search": (_tool("semantic_search", SEMANTIC_DESC, {
                "query": {"type": "string", "description": "Natural language query describing what information you're looking for"},
                "top_k": {**TOP_K_PROP, "description": "Number of most relevant results to return (default: 5, max: 20)"}},
                ["query"]), semantic_search),
            "read_chunk": (_tool("read_chunk", READ_DESC, {
                "chunk_ids": {"type": "array", "items": {"type": "string"},
                              "description": "List of chunk IDs to retrieve (e.g., ['0', '24', '172'])"}},
                ["chunk_ids"]), read_chunk),
        }
        ans, ctx, trace = agent.run(A_RAG_SYSTEM.format(style=self.style), q["question"], tools, self.max_loops, self.style)
        return Result(ans, ctx, trace, {"chunks_read": len(read)})
