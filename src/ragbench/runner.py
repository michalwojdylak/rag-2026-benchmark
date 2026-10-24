"""Run one method on one dataset: answer, score, and append to results/runs/<ds>/<method>.jsonl (resumable)."""

import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from ragbench import config, data, llm, metrics
from ragbench.methods import METHODS


def run_path(ds, method):
    return config.RUNS_DIR / ds / f"{method}.jsonl"


def load_runs(ds, method):
    p = run_path(ds, method)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.open()]


def _one(m, q, ds):
    usage = llm.Usage()
    token = llm.track(usage)
    t0 = time.time()
    try:
        res = m.answer(q)
        error = None
    except llm.BudgetExceeded:
        raise
    except Exception as e:
        res, error = None, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
    finally:
        llm.untrack(token)
    latency = time.time() - t0
    pred = res.answer if res else ""
    row = {
        "id": q["id"], "question": q["question"], "gold": q["answer"], "type": q.get("question_type", ""),
        "pred": pred, "error": error, "latency_s": round(latency, 2),
        # Latency without our response cache: recorded LLM time + measured retrieval time.
        "uncached_latency_s": round(usage.llm_seconds + usage.search_seconds, 2),
        "usage": usage.to_dict(),
        "n_contexts": len(res.contexts) if res else 0,
        "context_tokens_est": sum(len(c.split()) for c in res.contexts) * 4 // 3 if res else 0,
        "trace": res.trace if res else [],
        "extra": res.extra if res else {},
    }
    row["metrics"] = {
        "llm_acc": metrics.llm_judge(q["question"], pred, q["answer"]),
        "contain": metrics.contain(pred, q["answer"]),
        "em": metrics.exact_match(pred, q["answer"]),
        "f1": metrics.f1(pred, q["answer"]),
        "ctx_recall": metrics.context_recall(res.contexts, q["answer"]) if res and res.contexts else 0.0,
    }
    return row


def run(method, ds, n, workers=8):
    m = METHODS[method](ds)
    m.prepare()
    qs = data.questions(ds, n)
    done = {r["id"] for r in load_runs(ds, method)}
    todo = [q for q in qs if q["id"] not in done]
    path = run_path(ds, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not todo:
        return
    with path.open("a") as f, ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_one, m, q, ds) for q in todo]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{method}/{ds}"):
            row = fut.result()
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
