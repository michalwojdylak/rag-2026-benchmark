"""LightRAG (Guo et al., 2024; github.com/HKUDS/LightRAG, lightrag-hku) - the most used open-source GraphRAG.

Indexing: LightRAG's own entity/relation extraction with gpt-4o-mini over the corpus chunks,
bge-base embeddings, default local storages (NetworkX graph + NanoVectorDB).
Query: mode="hybrid" (local entity + global relation retrieval). We call LightRAG's own keyword
extraction prompt through our tracked client, pass the keywords in, take LightRAG's context string
(only_need_context) and answer with the shared answer prompt, so only retrieval differs.
"""

import asyncio
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ragbench import config, data, llm
from ragbench.embed import embed_docs, embed_query
from ragbench.methods.base import Method, Result, generate_answer

WORKDIR = config.DATA_PROCESSED / "lightrag"


class _Loop:
    """One event loop thread owns every LightRAG instance (LightRAG binds storages to a loop)."""
    _loop = None
    _lock = threading.Lock()

    @classmethod
    def run(cls, coro):
        with cls._lock:
            if cls._loop is None:
                cls._loop = asyncio.new_event_loop()
                # asyncio.to_thread uses this pool; the default (cpu+4 threads) caps LLM concurrency at ~12.
                cls._loop.set_default_executor(ThreadPoolExecutor(96))
                threading.Thread(target=cls._loop.run_forever, daemon=True).start()
        return asyncio.run_coroutine_threadsafe(coro, cls._loop).result()


class _EmbedBatcher:
    """LightRAG fires thousands of tiny embedding calls concurrently during graph merges; run whatever
    arrives within a few milliseconds as one GPU batch instead of one locked call each."""

    def __init__(self, window=0.02):
        self.window, self.pending, self.scheduled = window, [], False

    async def embed(self, texts):
        fut = asyncio.get_running_loop().create_future()
        self.pending.append((list(texts), fut))
        if not self.scheduled:
            self.scheduled = True
            asyncio.get_running_loop().call_later(self.window, lambda: asyncio.ensure_future(self._flush()))
        return await fut

    async def _flush(self):
        batch, self.pending, self.scheduled = self.pending, [], False
        flat = [t for texts, _ in batch for t in texts]
        try:
            emb = await asyncio.to_thread(embed_docs, flat, 64, False)
        except Exception as e:
            for _, fut in batch:
                fut.set_exception(e)
            return
        i = 0
        for texts, fut in batch:
            fut.set_result(emb[i:i + len(texts)])
            i += len(texts)


class LightRAGMethod(Method):
    name = "lightrag"
    family = "graph"
    reference = "LightRAG hybrid mode (lightrag-hku), gpt-4o-mini graph extraction"
    mode = "hybrid"
    chunks_per_doc = 64  # 2Wiki and Medical were built with 16; only commit frequency differs
    _instances = {}
    _build_lock = threading.Lock()
    index_usage = {}

    def _rag(self):
        with self._build_lock:
            if self.ds not in self._instances:
                self._instances[self.ds] = _Loop.run(self._create())
            return self._instances[self.ds]

    async def _create(self):
        from lightrag import LightRAG
        from lightrag.kg.shared_storage import initialize_pipeline_status
        from lightrag.utils import EmbeddingFunc

        usage = self.index_usage.setdefault(self.ds, llm.Usage())

        async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
                       (history_messages or []) + [{"role": "user", "content": prompt}]

            def call():
                token = llm.track(usage)
                try:
                    return llm.chat(messages, model=config.INDEXER)["content"]
                finally:
                    llm.untrack(token)
            return await asyncio.to_thread(call)

        batcher = _EmbedBatcher()

        async def embed_func(texts, context="document", **kwargs):
            if context == "query":
                return np.array([embed_query(t) for t in texts], dtype=np.float32)
            return await batcher.embed(texts)

        workdir = WORKDIR / self.ds
        workdir.mkdir(parents=True, exist_ok=True)
        rag = LightRAG(
            working_dir=str(workdir),
            workspace=self.ds,
            llm_model_func=llm_func,
            llm_model_name=config.INDEXER,
            llm_model_max_async=64,
            max_parallel_insert=48,
            # embeddings share one local GPU with other jobs; the 30 s default made vector upserts time out
            default_embedding_timeout=600,
            # let many single-text calls reach _EmbedBatcher at once so they share GPU batches
            embedding_func_max_async=256,
            embedding_batch_num=64,
            embedding_func=EmbeddingFunc(embedding_dim=config.EMBED_DIM, max_token_size=512, func=embed_func,
                                         supports_asymmetric=True),
        )
        await rag.initialize_storages()
        await initialize_pipeline_status(workspace=self.ds)
        return rag

    def prepare(self):
        rag = self._rag()
        done_marker = WORKDIR / self.ds / "ragbench_indexed.json"
        if done_marker.exists():
            return json.loads(done_marker.read_text())
        # LightRAG commits every store after each document; with its file-based default storages that
        # rewrites the whole graph per document (quadratic). Inserting 16 corpus chunks per document cuts
        # commits 16x; LightRAG re-chunks at 1,200 tokens anyway, so retrieval units are unchanged.
        texts = list(data.chunks(self.ds).values())
        docs = ["\n\n".join(texts[i:i + self.chunks_per_doc]) for i in range(0, len(texts), self.chunks_per_doc)]
        t0 = time.time()
        _Loop.run(rag.ainsert(docs, ids=[f"{self.ds}-doc-{i}" for i in range(len(docs))]))
        self._repair(rag, dict(zip([f"{self.ds}-doc-{i}" for i in range(len(docs))], docs)))
        usage = self.index_usage[self.ds]
        stats = {"index_seconds": round(time.time() - t0, 1), "documents": len(docs),
                 "failed_documents": len(self._failed_docs()), **usage.to_dict()}
        done_marker.write_text(json.dumps(stats, indent=2))
        return stats

    def _repair(self, rag, docs_by_id, rounds=3):
        """LightRAG leaves documents FAILED after e.g. a vector-store timeout and does not retry them;
        delete each failed one (dropping its partial graph) and insert its text again under a new id."""
        for r in range(rounds):
            failed = self._failed_docs()
            if not failed:
                return
            for doc_id in failed:
                base = doc_id.split("-retry")[0]
                _Loop.run(rag.adelete_by_doc_id(doc_id))
                _Loop.run(rag.ainsert(docs_by_id[base], ids=[f"{base}-retry{r}"]))

    def _failed_docs(self):
        path = WORKDIR / self.ds / self.ds / "kv_store_doc_status.json"
        status = json.loads(path.read_text()) if path.exists() else {}
        return [k for k, v in status.items() if not k.startswith("dup-") and v.get("status") != "processed"]

    def _keywords(self, question):
        from lightrag.prompt import PROMPTS
        prompt = PROMPTS["keywords_extraction"].format(
            query=question, examples="\n".join(PROMPTS["keywords_extraction_examples"]), language="English")
        out = llm.text(prompt)
        m = re.search(r"\{.*\}", out, re.DOTALL)
        try:
            kw = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            kw = {}
        return kw.get("high_level_keywords") or [question], kw.get("low_level_keywords") or [question]

    def answer(self, q):
        from lightrag import QueryParam
        rag = self._rag()
        hl, ll = self._keywords(q["question"])
        t0 = time.time()
        context = _Loop.run(rag.aquery(q["question"], param=QueryParam(
            mode=self.mode, only_need_context=True, hl_keywords=hl, ll_keywords=ll)))
        usage = llm.current()
        if usage is not None:
            usage.add_search(time.time() - t0)
        context = context or ""
        ans = generate_answer(q["question"], context, self.ds)
        return Result(ans, [context], [{"hl_keywords": hl, "ll_keywords": ll, "context_chars": len(context)}])
