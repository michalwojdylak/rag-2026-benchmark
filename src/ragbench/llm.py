"""OpenAI chat wrapper with per-question usage tracking, a disk cache and a global budget cap.

Every method calls `chat()`; the active `Usage` (set by the runner per question, or by the
indexer per build) collects calls, tokens, dollars and wall time, so cost numbers in the
results are measured, not estimated. Cached replies still count their original cost into
`Usage` (that is what the method costs) but not into the real-spend ledger.
"""

import contextvars
import hashlib
import json
import random
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict

from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError, InternalServerError

from ragbench import config


@dataclass
class Usage:
    llm_calls: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    llm_seconds: float = 0.0
    search_calls: int = 0
    search_seconds: float = 0.0
    by_model: dict = field(default_factory=dict)

    def add_search(self, seconds):
        self.search_calls += 1
        self.search_seconds += seconds

    def add(self, model, inp, cached, out, reasoning, cost, seconds):
        self.llm_calls += 1
        self.input_tokens += inp
        self.cached_tokens += cached
        self.output_tokens += out
        self.reasoning_tokens += reasoning
        self.cost_usd += cost
        self.llm_seconds += seconds
        m = self.by_model.setdefault(model, {"calls": 0, "cost_usd": 0.0})
        m["calls"] += 1
        m["cost_usd"] += cost

    def to_dict(self):
        return asdict(self)


_current = contextvars.ContextVar("usage", default=None)


def track(usage: Usage):
    """Make `usage` the sink for all chat() calls in this context. Returns a reset token."""
    return _current.set(usage)


def untrack(token):
    _current.reset(token)


def current():
    return _current.get()


class BudgetExceeded(RuntimeError):
    pass


def price(model, inp, cached, out):
    key = next((k for k in config.PRICES if model.startswith(k)), None)
    if key is None:
        return 0.0
    p_in, p_cached, p_out = config.PRICES[key]
    return ((inp - cached) * p_in + cached * p_cached + out * p_out) / 1e6


class _Cache:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS spend (id INTEGER PRIMARY KEY CHECK (id = 1), usd REAL)")
        self.db.execute("INSERT OR IGNORE INTO spend VALUES (1, 0)")
        self.db.commit()
        self.lock = threading.Lock()

    def get(self, k):
        with self.lock:
            row = self.db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k, v, cost):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (k, json.dumps(v)))
            self.db.execute("UPDATE spend SET usd = usd + ? WHERE id = 1", (cost,))
            self.db.commit()

    def spent(self):
        with self.lock:
            return self.db.execute("SELECT usd FROM spend WHERE id = 1").fetchone()[0]


_cache = None
_client = None
_init_lock = threading.Lock()


def _get_cache():
    global _cache
    with _init_lock:
        if _cache is None:
            _cache = _Cache(config.CACHE_DIR / "llm_cache.sqlite")
    return _cache


def _init():
    global _client
    cache = _get_cache()
    with _init_lock:
        if _client is None:
            _client = OpenAI(max_retries=0, timeout=180)
    return cache, _client


def spent_usd():
    return _get_cache().spent()


def chat(messages, model=None, tools=None, max_tokens=None, json_mode=False, reasoning=None, cache_tag=""):
    """Call the chat completions API (or Responses, for gpt-5 with reasoning). Returns the assistant message
    as a plain dict ({"role", "content", "tool_calls"?}). `cache_tag` lets a caller force distinct samples."""
    cache, client = _init()
    model = model or config.GENERATOR
    effort = reasoning or config.GENERATOR_REASONING
    if model.startswith("gpt-5") and effort != "none":
        # Chat Completions refuses function tools + reasoning for gpt-5.4, so reasoning runs use Responses.
        return _responses(cache, client, messages, model, tools, max_tokens, json_mode, effort, cache_tag)
    kwargs = {"model": model, "messages": messages}
    if model.startswith("gpt-5"):
        kwargs["reasoning_effort"] = effort
        if max_tokens:
            kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["temperature"] = 0
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
    if tools:
        kwargs["tools"] = tools
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    key = hashlib.sha256(json.dumps([kwargs, cache_tag], sort_keys=True).encode()).hexdigest()
    hit = cache.get(key)
    if hit is None:
        if cache.spent() > config.BUDGET_USD:
            raise BudgetExceeded(f"Real API spend passed ${config.BUDGET_USD}; raise RAGBENCH_BUDGET_USD to continue")
        t0 = time.time()
        for attempt in range(8):
            try:
                resp = client.chat.completions.create(**kwargs)
                break
            except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError):
                if attempt == 7:
                    raise
                time.sleep(min(60, 2 ** attempt + random.random()))
        seconds = time.time() - t0
        msg = resp.choices[0].message
        u = resp.usage
        details = getattr(u, "prompt_tokens_details", None)
        out_details = getattr(u, "completion_tokens_details", None)
        hit = {
            "message": {
                "role": "assistant",
                "content": msg.content or "",
                **({"tool_calls": [tc.model_dump() for tc in msg.tool_calls]} if msg.tool_calls else {}),
            },
            "usage": {
                "input": u.prompt_tokens,
                "cached": (details.cached_tokens or 0) if details else 0,
                "output": u.completion_tokens,
                "reasoning": (out_details.reasoning_tokens or 0) if out_details else 0,
            },
            "seconds": seconds,
        }
        cost = price(model, hit["usage"]["input"], hit["usage"]["cached"], hit["usage"]["output"])
        cache.put(key, hit, cost)

    _record(model, hit)
    return hit["message"]


def _record(model, hit):
    usage = _current.get()
    if usage is not None:
        h = hit["usage"]
        usage.add(model, h["input"], h["cached"], h["output"], h["reasoning"],
                  price(model, h["input"], h["cached"], h["output"]), hit["seconds"])


def _to_items(messages):
    """Chat-style messages -> Responses API input items. Assistant turns produced by `_responses` carry
    their raw output items (incl. encrypted reasoning) in `_items` and are replayed verbatim."""
    items = []
    for m in messages:
        if m["role"] == "assistant" and "_items" in m:
            items += m["_items"]
        elif m["role"] == "tool":
            items.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": m["content"]})
        elif m["role"] == "assistant":
            if m.get("content"):
                items.append({"role": "assistant", "content": m["content"]})
            for tc in m.get("tool_calls", []):
                items.append({"type": "function_call", "call_id": tc["id"], "name": tc["function"]["name"],
                              "arguments": tc["function"]["arguments"]})
        else:
            items.append({"role": m["role"], "content": m["content"]})
    return items


def _responses(cache, client, messages, model, tools, max_tokens, json_mode, effort, cache_tag):
    kwargs = {"model": model, "input": _to_items(messages), "reasoning": {"effort": effort},
              "store": False, "include": ["reasoning.encrypted_content"]}
    if tools:
        kwargs["tools"] = [{"type": "function", "name": t["function"]["name"],
                            "description": t["function"].get("description", ""),
                            "parameters": t["function"]["parameters"]} for t in tools]
    if max_tokens:
        kwargs["max_output_tokens"] = max_tokens
    if json_mode:
        kwargs["text"] = {"format": {"type": "json_object"}}
    key = hashlib.sha256(json.dumps([kwargs, cache_tag], sort_keys=True).encode()).hexdigest()
    hit = cache.get(key)
    if hit is None:
        if cache.spent() > config.BUDGET_USD:
            raise BudgetExceeded(f"Real API spend passed ${config.BUDGET_USD}; raise RAGBENCH_BUDGET_USD to continue")
        t0 = time.time()
        for attempt in range(8):
            try:
                resp = client.responses.create(**kwargs)
                break
            except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError):
                if attempt == 7:
                    raise
                time.sleep(min(60, 2 ** attempt + random.random()))
        out_items = [{k: v for k, v in it.model_dump(exclude_none=True).items() if k != "status"} for it in resp.output]
        calls = [it for it in out_items if it["type"] == "function_call"]
        u = resp.usage
        hit = {
            "message": {
                "role": "assistant",
                "content": resp.output_text or "",
                **({"tool_calls": [{"id": c["call_id"], "type": "function",
                                    "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]}
                   if calls else {}),
                "_items": out_items,
            },
            "usage": {
                "input": u.input_tokens,
                "cached": (u.input_tokens_details.cached_tokens or 0) if u.input_tokens_details else 0,
                "output": u.output_tokens,
                "reasoning": (u.output_tokens_details.reasoning_tokens or 0) if u.output_tokens_details else 0,
            },
            "seconds": time.time() - t0,
        }
        cache.put(key, hit, price(model, hit["usage"]["input"], hit["usage"]["cached"], hit["usage"]["output"]))
    _record(model, hit)
    return hit["message"]


def text(prompt, system=None, **kw):
    """Single-turn convenience wrapper returning only the reply text."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    return chat(messages, **kw)["content"].strip()
