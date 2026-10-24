"""MetaRAG - Metacognitive Retrieval-Augmented LLMs (Zhou et al., WWW 2024, arXiv 2402.11626).

Ported from the official code as used in the SIGIR 2026 reproducibility study
(github.com/iai-group/sigir2026-metarag). Loop: answer -> MONITOR (does an independent expert's
pseudo-answer agree?) -> if not, EVALUATE (internal/external knowledge judges + error checker) ->
PLAN (rewrite query and retrieve more, or adjust instructions) -> answer again, up to 3 iterations.

Substitutions (documented in the README): the T5-large expert QA model and T5-XXL NLI model are
replaced by a small LLM (gpt-5.4-nano) with the same role; the MiniLM similarity model is kept.
"""

import json
import re
import threading
from functools import lru_cache

from ragbench import config, llm, search
from ragbench.methods.base import Method, Result

# --- prompts, verbatim from metarag/config.py (few-shot examples kept, "ONE OR FEW WORDS" made dataset-aware) ---
QA_PROMPT = """Answer the question by thinking step by step. In your reasoning process, you need to carefully search for the information you need in the reference. If you need additional information, give the query you want to search.
Reference: {reference}
Question: {question}
{all_hint}
Your output MUST follow the following format:
```json
{{"reason": "your reasoning steps","answer": "your final answer {answer_len}"}}
```
Here are some examples:
#
Reference: John, Count Palatine of Neumarkt: John (Johann von Pfalz-Neumarkt; 1383 – 14 March 1443) was the Count Palatine of Neumarkt from 1410 to his death. The son of Rupert III of the Palatinate, he married Catherine of Pomerania in 1407.
Question: Who is Catherine Of Pomerania, Countess Palatine Of Neumarkt’s father-in-law?
Expected Output:
```json
{{"reason": "The husband of Catherine of Pomerania, Countess Palatine of Neumarkt is John, Count Palatine of Neumarkt. The father of John, Count Palatine of Neumarkt is Rupert III of the Palatinate. So the final answer is: Rupert III of the Palatinate.", "answer": "Rupert III of the Palatinate"}}
```
#
Reference: Crimen a las tres: Crimen a las tres is a 1935 Argentine crime film directed and written by Luis Saslavsky. Elio Petri: The Working Class Goes to Heaven (Italian: La classe operaia va in paradiso), released in the US as Lulu the Tool, is a 1971 political drama film directed by Elio Petri. Luis Saslavsky (April 21, 1903 – March 20, 1995) was an Argentine film director. Elio Petri: Final years. Petri died of cancer on 10 November 1982.
Question: Which film has the director died first, Crimen A Las Tres or The Working Class Goes To Heaven?
Expected Output:
```json
{{"reason": "The director of Crimen a las tres is Luis Saslavsky. The director of The Working Class Goes to Heaven is Elio Petri. Luis Saslavsky died on March 20, 1995. Elio Petri died on 10 November 1982. So the director of The Working Class Goes to Heaven died first.", "answer": "The Working Class Goes to Heaven"}}
```
"""

CHECK_PROMPT = """I need you to act as an inspector to check if there are any basic errors in the user's answer to the question. Below are some typical error types and corresponding examples.
Examples:
    1. Logic error
    question: What director worked with Vikram Bhatt on a film starring actors Rajneesh Duggal and Adah Sharma?
    user_answer: Vikram Bhatt
    ground truth: Tinu Suresh Desai
    explaination: The question is about someone who works with Vikram Bhatt, not Vikram Bhatt himself.
    2. Not align with the question's requirement
    question: The Waterloo Vase has been used as a garden ornament at whose residence and administrative headquarters?
    user_answer: Buckingham Palace
    ground truth: reigning monarch of the United Kingdom
    explaination: The question is about "whose residence", so the answer should refer to a person, not a place.
    3. Answer redundancy
    question: What type of film are both "500 Years Later" and "Manson"?
    user_answer: Documentary films
    ground truth: documentary
    explaination: The question is about the type of film, just answer the film type, no need to follow "films".
    4. Answer redundancy
    question: Who is older, Jack Ma or Faye Wong?
    user_answer: Jack Ma is older
    ground truth: Jack Ma
    explaination: Just answer the person's name, no additional explanation required.
Here is the user's answer that you need to check.
Question: {question}
user_answer: {answer}
Please check if the user's answer has made the above error and give your judgment and feedback. Your feedback can be similar to the explanation above.
Output in JSON format like:
```json
{{"judgement": "the error type (or correct)", "feedback": "some short reminders you would like to give"}}
```"""

REWRITE_PROMPT = """Question: {question}
Reference: {reference}
Given answer: {answer}
A person answered a question based on a reference information. However, due to issues with the reference (lack of information), more references need to be retrieved now. Please carefully consider his answers and questions step by step, and provide a query for searching based on the information he needs to answer the question.
Your output MUST with the ending of "The rewrite query is ..."
"""

EXTERNAL_PROMPT = "I need you to help me determine if the provided reference can answer the question, you just need to answer yes or no.\n\nQuestion: {question}Reference: {reference}\n\nYou just need to answer yes or no.\nAnswer:"
INTERNAL_PROMPT = "Determine if you can provide a reliable answer to the question: '{question}' based on your own knowledge(output yes or no)?"
# Replaces the T5 expert QA model: a small, independent model answers from the same references.
EXPERT_PROMPT = "Current Question: {question}\nSearch results: {reference}\nAnswer with a few words only.\nCurrent Answer:"
# Replaces the T5-XXL TRUE NLI model.
NLI_PROMPT = "Premise: {premise}\nHypothesis: {hypothesis}\nDoes the premise entail the hypothesis? Answer only 1 (yes) or 0 (no)."

CRITIC_SYSTEM = "Act as an Evaluator & Critic System."
THRESHOLD = 0.3   # official default
MAX_ITER = 3      # official default
REF_NUM = 5       # official default

_sim_lock = threading.Lock()


@lru_cache(maxsize=1)
def _sim_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


def similarity(a, b):
    with _sim_lock:
        e = _sim_model().encode([a, b], normalize_embeddings=True)
    return float(e[0] @ e[1])


def _parse_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


class MetaRAG(Method):
    name = "metarag"
    family = "metacognitive"
    reference = "MetaRAG (WWW'24) monitor-evaluate-plan loop, official prompts"

    def __init__(self, ds):
        super().__init__(ds)
        self.answer_len = "in ONE OR FEW WORDS" if self.short else "in one to three sentences"

    def _retrieve(self, query):
        return [h["text"] for h in search.hybrid(f"{self.ds}-passages", query, k=REF_NUM)]

    def _generate(self, question, refs, suggestion=None, hint=None):
        all_hint = (f"Here are some suggestions you need to follow: {suggestion}\n" if suggestion else "") + \
                   (f"Here are some mistakes you may make, you need to be careful: {hint}\n" if hint else "")
        out = llm.text(QA_PROMPT.format(reference=" ".join(refs), question=question, all_hint=all_hint,
                                        answer_len=self.answer_len), system="Act as an LLM helper.")
        parsed = _parse_json(out)
        if parsed and "answer" in parsed:
            return str(parsed.get("reason", "")), str(parsed["answer"])
        return out, out

    def _check_answer(self, answer):
        if any(k in answer.casefold() for k in ("can't determin", "unknown", "cannot determin")):
            return False
        return not self.short or len(answer.split()) <= 10

    def answer(self, q):
        question = q["question"]
        refs = self._retrieve(question)
        all_refs = list(refs)
        reason, ans = self._generate(question, refs)
        final, logs = ans, []
        while True:
            pseudo = llm.text(EXPERT_PROMPT.format(question=question, reference=" ".join(refs)), model=config.HELPER)
            score = similarity(pseudo, ans)
            log = {"answer": ans, "expert": pseudo, "monitor_score": round(score, 3)}
            if score > THRESHOLD or len(logs) >= MAX_ITER:
                logs.append(log)
                final = ans
                break
            check = _parse_json(llm.text(CHECK_PROMPT.format(question=question, answer=ans), system=CRITIC_SYSTEM)) \
                or {"judgement": "correct", "feedback": ""}
            internal = llm.text(INTERNAL_PROMPT.format(question=question), system=CRITIC_SYSTEM).lower()
            internal_ok = not any(k in internal for k in ("sorry", "couldn't", "don't have access", "no"))
            external_ok = "no" not in llm.text(EXTERNAL_PROMPT.format(question=question, reference="\n".join(refs)),
                                               system=CRITIC_SYSTEM).lower()
            hint = "Please think step by step." if str(check.get("judgement", "")).lower() == "correct" else check.get("feedback")
            need_more = False
            if internal_ok and external_ok:
                suggestion = "Carefully check your final answer. Do not add irrelevant content and answer questions accurately."
                nli = llm.text(NLI_PROMPT.format(premise=reason, hypothesis=ans), model=config.HELPER)
                if not nli.strip().startswith("1"):
                    suggestion = "Please try to provide an answer by considering multi-step reasoning."
            elif internal_ok:
                suggestion = "If you feel that there is no suitable content in the reference, try relying on yourself to answer the question."
                need_more = True
            elif external_ok:
                suggestion = "You need to answer questions entirely based on reference, and do not use your own knowledge."
            else:
                suggestion = "You can break down the question into sub questions to use reference."
                need_more = True
            if need_more:
                rw = llm.text(REWRITE_PROMPT.format(question=question, answer=ans, reference="\n".join(refs)))
                new_query = re.split(r"the rewrite query is", rw, flags=re.IGNORECASE)[-1].strip(" :.\"'\n")[:64] or question
                new = self._retrieve(new_query)
                refs = list(dict.fromkeys(refs + new))
                all_refs += new
                log["rewrite_query"] = new_query
            log.update(internal=internal_ok, external=external_ok, judgement=check.get("judgement"), suggestion=suggestion)
            logs.append(log)
            reason, new_ans = self._generate(question, refs, suggestion, hint)
            if self._check_answer(new_ans):
                final = new_ans
            ans = new_ans
        return Result(final, all_refs, logs, {"iterations": len(logs)})
