"""Search-R1 (Jin et al., 2025, arXiv 2503.09516): a 7B model trained with RL (PPO) to interleave
<think>, <search> and <answer>. Stands in for the "R1 search agent" family (R1-Searcher, ReSearch,
Agentic-RAG-R1, ...) - it is the one with public, widely used checkpoints.

Checkpoint: PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo (Q4_K_M GGUF by mradermacher),
served locally by llama.cpp `llama-server` on Apple Metal. Prompt and stop logic follow the
official infer.py; the retriever is our OpenSearch hybrid search (top-3, as in the paper).
"""

import os
import re
import time

import requests

from ragbench import llm, search
from ragbench.methods.base import Method, Result, short_trace

URL = os.getenv("SEARCH_R1_URL", "http://127.0.0.1:8081")

PROMPT = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

# Qwen2.5 chat template with its default system prompt (what tokenizer.apply_chat_template produces).
CHAT = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")

MAX_SEARCHES = 6
TOP_K = 3


def complete(prompt, stop, n_predict=1024):
    t0 = time.time()
    r = requests.post(f"{URL}/completion", json={
        "prompt": prompt, "n_predict": n_predict, "temperature": 0, "stop": stop, "cache_prompt": True,
    }, timeout=900).json()
    usage = llm.current()
    if usage is not None:  # local model: tokens and time are real, dollars are zero
        usage.add("local:search-r1-7b", r.get("tokens_evaluated", 0), 0, r.get("tokens_predicted", 0), 0, 0.0, time.time() - t0)
    return r["content"], r.get("stopping_word", "")


class SearchR1(Method):
    name = "search_r1"
    family = "rl-trained"
    reference = "Search-R1 7B (PPO) local via llama.cpp, OpenSearch retriever"

    def _search(self, query):
        hits = search.hybrid(f"{self.ds}-passages", query, k=TOP_K)
        text = "".join(f"Doc {i + 1}(Title: {' '.join(h['text'].split()[:6])}) {h['text']}\n" for i, h in enumerate(hits))
        return text, [h["text"] for h in hits]

    def answer(self, q):
        question = q["question"].strip()
        if not question.endswith("?"):
            question += "?"
        prompt = CHAT.format(prompt=PROMPT.format(question=question))
        contexts, trace = [], []
        for _ in range(MAX_SEARCHES + 1):
            out, stop = complete(prompt, ["</search>", "</answer>"])
            if stop == "</answer>" or "<answer>" in out:
                out += "</answer>"
                prompt += out
                break
            m = re.findall(r"<search>(.*)", out, re.DOTALL)
            if not m or len(trace) >= MAX_SEARCHES:
                prompt += out
                break
            query = m[-1].strip()
            info, ctx = self._search(query)
            contexts += ctx
            trace.append({"query": query, "result": short_trace(info)})
            prompt += out + "</search>" + f"\n\n<information>{info}</information>\n\n"
        answers = re.findall(r"<answer>(.*?)</answer>", prompt.split("<|im_start|>assistant\n", 1)[1], re.DOTALL)
        return Result(answers[-1].strip() if answers else "", contexts, trace)
