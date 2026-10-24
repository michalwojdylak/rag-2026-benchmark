"""Answer and retrieval metrics.

EM / F1      SQuAD-style normalized exact match and token F1 (short-answer datasets only)
Contain      gold answer (normalized) is a substring of the prediction - A-RAG's "Cont-Acc"
LLM-Acc      an LLM judge decides correct/incorrect against the gold answer - A-RAG's "LLM-Acc"
Ctx-Recall   gold answer (normalized) appears anywhere in the corpus text the method showed the LLM
"""

import re
import string
from collections import Counter

import numpy as np

from ragbench import config, llm


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, gold):
    return float(normalize(pred) == normalize(gold))


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def contain(pred, gold):
    g = normalize(gold)
    return float(bool(g) and g in normalize(pred))


def context_recall(contexts, gold):
    g = normalize(gold)
    return float(bool(g) and any(g in normalize(c) for c in contexts))


# Same rubric as A-RAG's evaluator, plus the question so the judge can resolve what is being asked.
JUDGE_PROMPT = """Please evaluate if the generated answer is correct by comparing it with the gold answer.
Question: {question}
Generated answer: {pred}
Gold answer: {gold}

The generated answer should be considered correct if it:
1. Contains the key information from the gold answer
2. Is factually accurate and consistent with the gold answer
3. Does not contain any contradicting information

Respond with ONLY 'correct' or 'incorrect'.
Response:"""


def llm_judge(question, pred, gold):
    if not pred.strip():
        return 0.0
    out = llm.text(JUDGE_PROMPT.format(question=question, pred=pred, gold=gold),
                   system="You are an expert evaluator.", model=config.JUDGE, max_tokens=5)
    return float(out.strip().lower().startswith("correct"))


def bootstrap_ci(values, n=2000, seed=0):
    """95% percentile bootstrap CI of the mean."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
