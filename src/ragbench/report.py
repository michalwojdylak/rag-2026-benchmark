"""Aggregate results/runs/*/*.jsonl into results/summary.{csv,md} and results/figures/*.png."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ragbench import config, metrics  # noqa: E402
from ragbench.methods import METHODS  # noqa: E402

FIG = config.RESULTS_DIR / "figures"
ORDER = list(METHODS)
DS_LABEL = {"musique": "MuSiQue", "hotpotqa": "HotpotQA", "2wikimultihop": "2Wiki", "medical": "Medical", "novel": "Novel"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


REASONING_DIR = config.RESULTS_DIR / "runs_reasoning_medium"


def load(runs_dir=config.RUNS_DIR):
    rows = []
    for ds in config.DATASETS:
        for path in sorted((runs_dir / ds).glob("*.jsonl")) if (runs_dir / ds).exists() else []:
            for line in path.open():
                r = json.loads(line)
                u = r["usage"]
                rows.append({"dataset": ds, "method": path.stem, "id": r["id"], "type": r["type"], "error": bool(r["error"]),
                             **r["metrics"], "cost_usd": u["cost_usd"], "llm_calls": u["llm_calls"],
                             "search_calls": u["search_calls"], "input_tokens": u["input_tokens"],
                             "output_tokens": u["output_tokens"], "latency_s": r["uncached_latency_s"]})
    return pd.DataFrame(rows)


def summarize(df):
    out = []
    for (m, ds), g in df.groupby(["method", "dataset"]):
        lo, hi = metrics.bootstrap_ci(g["llm_acc"])
        out.append({"method": m, "dataset": ds, "n": len(g), "llm_acc": g["llm_acc"].mean(), "llm_acc_lo": lo,
                    "llm_acc_hi": hi, "contain": g["contain"].mean(), "em": g["em"].mean(), "f1": g["f1"].mean(),
                    "ctx_recall": g["ctx_recall"].mean(), "cost_per_q": g["cost_usd"].mean(),
                    "llm_calls": g["llm_calls"].mean(), "search_calls": g["search_calls"].mean(),
                    "input_tokens": g["input_tokens"].mean(), "output_tokens": g["output_tokens"].mean(),
                    "latency_p50": g["latency_s"].median(), "errors": int(g["error"].sum())})
    return pd.DataFrame(out)


def _lightrag_calls(ds):
    path = config.DATA_PROCESSED / "lightrag" / ds / ds / "kv_store_llm_response_cache.json"
    return len(json.loads(path.read_text())) if path.exists() else 0


def index_costs():
    """Graph-R1 costs are measured. LightRAG builds were restarted (its own LLM cache then skips calls our
    tracker would see), so its cost is LightRAG's cached-call count x the per-call cost measured on the
    one uninterrupted build (Medical)."""
    rows = []
    med = config.DATA_PROCESSED / "lightrag" / "medical" / "ragbench_indexed.json"
    per_call = json.loads(med.read_text())["cost_usd"] / _lightrag_calls("medical") if med.exists() else float("nan")
    for ds in config.DATASETS:
        path = config.DATA_PROCESSED / ds / "graph_r1_index.json"
        if path.exists():
            s = json.loads(path.read_text())
            rows.append({"method": "graph_r1", "dataset": ds, "index_cost_usd": s["cost_usd"], "llm_calls": s["llm_calls"],
                         "estimated": False})
        if (config.DATA_PROCESSED / "lightrag" / ds / "ragbench_indexed.json").exists():
            calls = _lightrag_calls(ds)
            rows.append({"method": "lightrag", "dataset": ds, "index_cost_usd": calls * per_call, "llm_calls": calls,
                         "estimated": ds != "medical"})
    return pd.DataFrame(rows)


def _pct(x):
    return f"{100 * x:.1f}"


def markdown(summary, idx):
    methods = [m for m in ORDER if m in set(summary.method)]
    dss = [d for d in config.DATASETS if d in set(summary.dataset)]
    lines = ["## LLM-Acc (%) by dataset, 95% bootstrap CI in brackets", "",
             "| Method | " + " | ".join(DS_LABEL[d] for d in dss) + " | Multi-hop avg | All-5 avg |", "|---" * (len(dss) + 3) + "|"]
    for m in methods:
        cells = []
        for d in dss:
            r = summary[(summary.method == m) & (summary.dataset == d)]
            cells.append(f"{_pct(r.llm_acc.iloc[0])} [{_pct(r.llm_acc_lo.iloc[0])}–{_pct(r.llm_acc_hi.iloc[0])}]" if len(r) else "–")
        g = summary[summary.method == m]
        mh = g[g.dataset.isin(config.SHORT_ANSWER)]
        mh_avg = f"**{_pct(mh.llm_acc.mean())}**" if len(mh) == len(config.SHORT_ANSWER) else "–"
        all_avg = _pct(g.llm_acc.mean()) if len(g) == len(config.DATASETS) else "–"
        lines.append(f"| `{m}` | " + " | ".join(cells) + f" | {mh_avg} | {all_avg} |")
    short = summary[summary.dataset.isin(config.SHORT_ANSWER)]
    lines += ["", "## Short-answer metrics, averaged over MuSiQue / HotpotQA / 2Wiki (%)", "",
              "| Method | LLM-Acc | Contain | EM | F1 | Ctx-Recall |", "|---|---|---|---|---|---|"]
    for m in methods:
        g = short[short.method == m]
        if len(g):
            lines.append(f"| `{m}` | {_pct(g.llm_acc.mean())} | {_pct(g.contain.mean())} | {_pct(g.em.mean())} | "
                         f"{_pct(g.f1.mean())} | {_pct(g.ctx_recall.mean())} |")
    lines += ["", "## Cost and effort per question (mean over all datasets)", "",
              "| Method | $ / 1k questions | LLM calls | Searches | Input tok | Output tok | p50 latency (s) | Errors |",
              "|---|---|---|---|---|---|---|---|"]
    for m in methods:
        g = summary[summary.method == m]
        lines.append(f"| `{m}` | {1000 * g.cost_per_q.mean():.2f} | {g.llm_calls.mean():.1f} | {g.search_calls.mean():.1f} | "
                     f"{g.input_tokens.mean():,.0f} | {g.output_tokens.mean():,.0f} | {g.latency_p50.median():.1f} | {int(g.errors.sum())} |")
    if len(idx):
        lines += ["", "## One-off graph indexing cost (gpt-4o-mini)", "", "| Method | Dataset | LLM calls | $ |", "|---|---|---|---|"]
        for _, r in idx.iterrows():
            lines.append(f"| `{r.method}` | {DS_LABEL[r.dataset]} | {r.llm_calls:,} | {r.index_cost_usd:.2f}{' (est.)' if r.estimated else ''} |")
    return "\n".join(lines) + "\n"


def reasoning_table(df):
    """Paired comparison on the same questions: reasoning_effort none (main run) vs medium."""
    med = load(REASONING_DIR)
    if med.empty:
        return ""
    keys = ["dataset", "method", "id"]
    both = med.merge(df, on=keys, suffixes=("_med", "_none"))
    lines = ["", "## Experiment 2: same methods with `reasoning_effort=\"medium\"` (paired, same questions)", "",
             "| Method | Dataset | n | LLM-Acc none | LLM-Acc medium | Δ (pp) | $ / 1k none | $ / 1k medium |",
             "|---|---|---|---|---|---|---|---|"]
    for (m, ds), g in both.groupby(["method", "dataset"], sort=False):
        if m not in ORDER:
            continue
        a, b = g.llm_acc_none.mean(), g.llm_acc_med.mean()
        lines.append(f"| `{m}` | {DS_LABEL[ds]} | {len(g)} | {_pct(a)} | {_pct(b)} | {100 * (b - a):+.1f} | "
                     f"{1000 * g.cost_usd_none.mean():.2f} | {1000 * g.cost_usd_med.mean():.2f} |")
    order = {m: i for i, m in enumerate(ORDER)}
    head, rows = lines[:5], sorted(lines[5:], key=lambda l: (order[l.split("`")[1]], l))
    return "\n".join(head + rows) + "\n"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_frontier(summary):
    mh = summary[summary.dataset.isin(config.SHORT_ANSWER)]
    complete = [m for m in ORDER if (mh.method == m).sum() == len(config.SHORT_ANSWER)]
    g = mh.groupby("method").agg(acc=("llm_acc", "mean"), cost=("cost_per_q", "mean")).reindex(complete)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    _style(ax)
    x = 1000 * g.cost.clip(lower=1e-5)  # local Search-R1 costs $0; draw it at $0.01 / 1k
    ax.scatter(x, 100 * g.acc, s=70, color="#2a78d6", edgecolor=SURFACE, linewidth=2, zorder=3)
    below = {"a_rag", "naive_rag"}  # neighbours of agentic_rag / rerank_rag; label under the point
    for m, xi, yi in zip(g.index, x, 100 * g.acc):
        ax.annotate(m, (xi, yi), xytext=(7, -13 if m in below else 4), textcoords="offset points", fontsize=9, color=INK)
    ax.set_xscale("log")
    ticks = [t for t in (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30) if x.min() / 2 <= t <= x.max() * 2]
    ax.set_xticks(ticks, [f"${t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("API cost per 1,000 questions (USD, log scale; local Search-R1 is \\$0, drawn at \\$0.01)", color=INK2, fontsize=9)
    ax.set_ylabel("LLM-Acc, mean of MuSiQue / HotpotQA / 2Wiki (%)", color=INK2, fontsize=9)
    ax.set_title("Accuracy vs. cost (multi-hop datasets, reasoning off)", loc="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "accuracy_vs_cost.png", dpi=160)
    plt.close(fig)


def plot_heatmap(summary):
    methods = [m for m in ORDER if m in set(summary.method)]
    dss = [d for d in config.DATASETS if d in set(summary.dataset)]
    mat = np.full((len(methods), len(dss)), np.nan)
    for i, m in enumerate(methods):
        for j, d in enumerate(dss):
            r = summary[(summary.method == m) & (summary.dataset == d)]
            if len(r):
                mat[i, j] = 100 * r.llm_acc.iloc[0]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(methods) + 1.4), facecolor=SURFACE)
    ax.imshow(mat, cmap=cmap, vmin=np.nanmin(mat), vmax=np.nanmax(mat), aspect="auto")
    ax.set_xticks(range(len(dss)), [DS_LABEL[d] for d in dss], color=INK2, fontsize=9)
    ax.set_yticks(range(len(methods)), methods, color=INK, fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    mid = (np.nanmin(mat) + np.nanmax(mat)) / 2
    for i in range(len(methods)):
        for j in range(len(dss)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=9,
                        color="#ffffff" if mat[i, j] > mid else INK)
    ax.set_xticks(np.arange(-0.5, len(dss)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(methods)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.set_title("LLM-Acc (%) by method and dataset", loc="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "accuracy_heatmap.png", dpi=160)
    plt.close(fig)


def plot_reasoning(df):
    """Dumbbell: LLM-Acc with reasoning off vs medium, paired questions, MuSiQue + 2Wiki pooled."""
    med = load(REASONING_DIR)
    if med.empty:
        return
    both = med.merge(df, on=["dataset", "method", "id"], suffixes=("_med", "_none"))
    g = both.groupby("method")[["llm_acc_none", "llm_acc_med"]].mean().reindex([m for m in ORDER if m in set(both.method)])
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(g) + 1.5), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(g))[::-1]
    for yi, (a, b) in zip(y, zip(100 * g.llm_acc_none, 100 * g.llm_acc_med)):
        ax.plot([a, b], [yi, yi], color=GRID, linewidth=2, zorder=1)
    ax.scatter(100 * g.llm_acc_none, y, s=70, color="#86b6ef", edgecolor=SURFACE, linewidth=2, zorder=3, label="reasoning off")
    ax.scatter(100 * g.llm_acc_med, y, s=70, color="#184f95", edgecolor=SURFACE, linewidth=2, zorder=3, label="reasoning medium")
    for yi, b in zip(y, 100 * g.llm_acc_med):
        ax.annotate(f"{b:.0f}", (b, yi), xytext=(8, -3), textcoords="offset points", fontsize=9, color=INK)
    ax.set_yticks(y, g.index, color=INK, fontsize=9)
    ax.set_xlabel("LLM-Acc on the same 200 questions (100 MuSiQue + 100 2Wiki), %", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK2)
    ax.set_title("Turning reasoning on (gpt-5.4-mini)", loc="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "reasoning_effect.png", dpi=160)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = load()
    if df.empty:
        print("no runs yet")
        return
    summary = summarize(df)
    idx = index_costs()
    summary.to_csv(config.RESULTS_DIR / "summary.csv", index=False)
    by_type = df.groupby(["method", "dataset", "type"]).llm_acc.agg(["mean", "count"]).reset_index()
    by_type.to_csv(config.RESULTS_DIR / "by_question_type.csv", index=False)
    if len(idx):
        idx.to_csv(config.RESULTS_DIR / "index_costs.csv", index=False)
    md = markdown(summary, idx) + reasoning_table(df)
    (config.RESULTS_DIR / "summary.md").write_text(md)
    plot_frontier(summary)
    plot_heatmap(summary)
    plot_reasoning(df)
    _inject_readme(md)
    print(md)


def _inject_readme(md):
    """Keep the README tables identical to results/summary.md."""
    readme = config.ROOT / "README.md"
    text = readme.read_text()
    start, end = "<!-- RESULTS:START (generated by `ragbench report`) -->", "<!-- RESULTS:END -->"
    if start in text and end in text:
        head, rest = text.split(start, 1)
        readme.write_text(head + start + "\n\n" + md.replace("## ", "### ") + "\n" + end + rest.split(end, 1)[1])
