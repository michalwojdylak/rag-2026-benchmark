"""Reference points: no retrieval, one-shot hybrid retrieval, and hybrid retrieval + cross-encoder."""

from ragbench import llm, search
from ragbench.embed import rerank
from ragbench.methods.base import Method, Result, answer_style, format_passages, generate_answer


class ClosedBook(Method):
    name = "closed_book"
    family = "baseline"
    reference = "LLM only, no retrieval"

    def answer(self, q):
        return Result(llm.text(f"Question: {q['question']}\n\n{answer_style(self.ds)}"))


class NaiveRAG(Method):
    name = "naive_rag"
    family = "baseline"
    reference = "OpenSearch hybrid (BM25 + k-NN, RRF) top-k -> LLM"
    k = 8

    def retrieve(self, question):
        return search.hybrid(f"{self.ds}-passages", question, k=self.k)

    def answer(self, q):
        hits = self.retrieve(q["question"])
        ans = generate_answer(q["question"], format_passages(hits), self.ds)
        return Result(ans, [h["text"] for h in hits], [{"retrieved": [h["id"] for h in hits]}])


class RerankRAG(NaiveRAG):
    name = "rerank_rag"
    reference = "Hybrid top-30 -> bge-reranker-v2-m3 -> top-k -> LLM"
    candidates = 30

    def retrieve(self, question):
        hits = search.hybrid(f"{self.ds}-passages", question, k=self.candidates)
        scores = rerank(question, [h["text"] for h in hits])
        ranked = [h for _, h in sorted(zip(scores, hits), key=lambda x: -x[0])]
        return ranked[: self.k]
