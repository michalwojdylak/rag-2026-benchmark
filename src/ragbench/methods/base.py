"""Common method interface and the answer prompt shared by the non-agentic methods."""

from dataclasses import dataclass, field

from ragbench import config, llm


@dataclass
class Result:
    answer: str
    contexts: list = field(default_factory=list)   # every corpus text shown to the LLM
    trace: list = field(default_factory=list)      # steps for qualitative inspection
    extra: dict = field(default_factory=dict)


class Method:
    name = "base"
    family = ""
    reference = ""

    def __init__(self, ds):
        self.ds = ds
        self.short = ds in config.SHORT_ANSWER

    def prepare(self):
        """Index-time work beyond the shared OpenSearch indexes (e.g. building a graph)."""

    def answer(self, q) -> Result:
        raise NotImplementedError

    @property
    def style(self):
        return answer_style(self.ds)


def answer_style(ds):
    if ds in config.SHORT_ANSWER:
        return "Give only the final answer as a short phrase (an entity, date, number or yes/no), with no explanation."
    return "Give the final answer in one to three sentences."


ANSWER_PROMPT = """Answer the question using the context passages.

Context:
{context}

Question: {question}

{style}"""


def format_passages(hits, key="text"):
    return "\n\n".join(f"[{i + 1}] {h[key]}" for i, h in enumerate(hits))


def generate_answer(question, context, ds):
    return llm.text(ANSWER_PROMPT.format(context=context, question=question, style=answer_style(ds)))


def short_trace(text, n=300):
    return text if len(text) <= n else text[:n] + "..."
