"""OpenSearch: the single search engine behind every method.

Indexes per dataset:
  {ds}-passages   ~150-word passages, BM25 text + dense vector  (naive/rerank/agentic/Search-R1/MetaRAG)
  {ds}-sentences  single sentences + dense vector                 (A-RAG semantic_search)
  {ds}-chunks     original chunks, BM25 + phrase highlighting      (A-RAG keyword_search / read_chunk)
  {ds}-g1-*       Graph-R1 hypergraph: entities and hyperedges     (see methods/graph_r1.py)
"""

import time
from functools import lru_cache

from opensearchpy import OpenSearch, helpers

from ragbench import config, llm
from ragbench.embed import embed_query

RRF_PIPELINE = "ragbench-rrf"


@lru_cache(maxsize=1)
def client():
    return OpenSearch(config.OPENSEARCH_URL, timeout=120, pool_maxsize=32)


def create_index(name, vector=True, extra_props=None, recreate=True):
    os_ = client()
    if os_.indices.exists(index=name):
        if not recreate:
            return
        os_.indices.delete(index=name)
    props = {"text": {"type": "text", "analyzer": "english"}, "chunk_id": {"type": "keyword"}}
    if vector:
        props["embedding"] = {
            "type": "knn_vector", "dimension": config.EMBED_DIM,
            "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil",
                       "parameters": {"m": 16, "ef_construction": 128}},
        }
    props.update(extra_props or {})
    os_.indices.create(index=name, body={
        "settings": {"index": {"knn": vector, "number_of_shards": 1, "number_of_replicas": 0,
                               "refresh_interval": "-1"}},
        "mappings": {"properties": props},
    })


def bulk_index(name, docs):
    """docs: iterable of dicts with an 'id' key; everything else is stored as the source."""
    actions = ({"_index": name, "_id": d["id"], "_source": {k: v for k, v in d.items() if k != "id"}} for d in docs)
    helpers.bulk(client(), actions, chunk_size=500, request_timeout=300)
    client().indices.put_settings(index=name, body={"index": {"refresh_interval": "1s"}})
    client().indices.refresh(index=name)


def ensure_pipeline():
    client().transport.perform_request("PUT", f"/_search/pipeline/{RRF_PIPELINE}", body={
        "description": "Reciprocal rank fusion of BM25 and k-NN sub-queries",
        "phase_results_processors": [
            {"score-ranker-processor": {"combination": {"technique": "rrf", "rank_constant": 60}}}],
    })


def _search(index, body, params=None):
    t0 = time.time()
    res = client().search(index=index, body=body, params=params or {})
    usage = llm.current()
    if usage is not None:
        usage.add_search(time.time() - t0)
    return [{"id": h["_id"], "score": h["_score"], **h["_source"], **({"highlight": h["highlight"]} if "highlight" in h else {})}
            for h in res["hits"]["hits"]]


_NO_VEC = {"excludes": ["embedding"]}


def bm25(index, query, k=config.TOP_K):
    return _search(index, {"size": k, "_source": _NO_VEC, "query": {"match": {"text": query}}})


def knn(index, query, k=config.TOP_K, filter_=None):
    q = {"vector": embed_query(query), "k": k}
    if filter_:
        q["filter"] = filter_
    return _search(index, {"size": k, "_source": _NO_VEC, "query": {"knn": {"embedding": q}}})


def hybrid(index, query, k=config.TOP_K):
    """OpenSearch native hybrid query: BM25 + k-NN fused with RRF in a search pipeline."""
    body = {
        "size": k, "_source": _NO_VEC,
        "query": {"hybrid": {"queries": [
            {"match": {"text": query}},
            {"knn": {"embedding": {"vector": embed_query(query), "k": max(k, 20)}}},
        ]}},
    }
    return _search(index, body, params={"search_pipeline": RRF_PIPELINE})


def phrase_search(index, phrases, k=config.TOP_K):
    """Any-of exact phrase match with highlighted sentence fragments (A-RAG keyword search)."""
    body = {
        "size": k, "_source": _NO_VEC,
        "query": {"bool": {"should": [{"match_phrase": {"text": p}} for p in phrases], "minimum_should_match": 1}},
        "highlight": {"fields": {"text": {"type": "unified", "fragment_size": 220, "number_of_fragments": 5,
                                          "pre_tags": [""], "post_tags": [""]}}},
    }
    return _search(index, body)


def get_docs(index, ids):
    if not ids:
        return []
    res = client().mget(index=index, body={"ids": list(ids)}, params={"_source_excludes": "embedding"})
    return [{"id": d["_id"], **d["_source"]} for d in res["docs"] if d.get("found")]


def count(index):
    if not client().indices.exists(index=index):
        return 0
    return client().count(index=index)["count"]
