"""Print side-by-side traces for questions where methods disagree (material for the write-up).

uv run python scripts/examples.py 2wikimultihop a_rag            # none vs medium reasoning
uv run python scripts/examples.py musique naive_rag agentic_rag  # two methods, main run
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results"


def load(runs, ds, method):
    p = ROOT / runs / ds / f"{method}.jsonl"
    return {r["id"]: r for r in map(json.loads, p.open())} if p.exists() else {}


def show(r, label):
    print(f"  [{label}] pred={r['pred'][:120]!r} acc={r['metrics']['llm_acc']} calls={r['usage']['llm_calls']}")
    for t in r["trace"][:8]:
        if "tool" in t:
            print(f"      {t['tool']}({json.dumps(t['args'])[:100]})")
        elif "query" in t:
            print(f"      query: {t['query'][:100]}")


def main():
    ds, m1 = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3:
        a, b, la, lb = load("runs", ds, m1), load("runs", ds, sys.argv[3]), m1, sys.argv[3]
    else:
        a, b, la, lb = load("runs", ds, m1), load("runs_reasoning_medium", ds, m1), "none", "medium"
    flips = [i for i in b if i in a and a[i]["metrics"]["llm_acc"] != b[i]["metrics"]["llm_acc"]]
    print(f"{len(flips)} disagreements out of {len(set(a) & set(b))} shared questions\n")
    for i in flips[:6]:
        print("Q:", a[i]["question"], "| gold:", a[i]["gold"])
        show(a[i], la)
        show(b[i], lb)
        print()


if __name__ == "__main__":
    main()
