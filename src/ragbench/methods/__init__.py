from ragbench.methods.agentic import ARAG, AgenticRAG
from ragbench.methods.baselines import ClosedBook, NaiveRAG, RerankRAG
from ragbench.methods.graph_r1 import GraphR1, GraphR1Tools
from ragbench.methods.lightrag_method import LightRAGMethod
from ragbench.methods.metarag import MetaRAG
from ragbench.methods.search_r1 import SearchR1

METHODS = {m.name: m for m in [ClosedBook, NaiveRAG, RerankRAG, MetaRAG, AgenticRAG, ARAG, SearchR1, LightRAGMethod, GraphR1, GraphR1Tools]}
