"""A minimal function-calling agent loop (the same loop A-RAG ships), shared by the tool-using methods."""

import json

from ragbench import llm
from ragbench.methods.base import short_trace

FORCE_PROMPT = ("You have reached the limit. You MUST now provide a final answer based on the information "
                "you have gathered so far. Do NOT call any more tools. {style}")


def run(system, user, tools, max_loops, style):
    """tools: {name: (schema, fn(**args) -> (text, contexts))}. Returns (answer, contexts, trace)."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    schemas = [schema for schema, _ in tools.values()]
    contexts, trace = [], []
    for loop in range(max_loops):
        msg = llm.chat(messages, tools=schemas)
        messages.append(msg)
        calls = msg.get("tool_calls")
        if not calls:
            return msg["content"], contexts, trace
        for tc in calls:
            name, args = tc["function"]["name"], {}
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
                out, ctx = tools[name][1](**args)
            except Exception as e:  # a bad tool call is the agent's problem, not the run's
                out, ctx = f"Error executing tool: {e}", []
            contexts += ctx
            trace.append({"loop": loop + 1, "tool": name, "args": args, "result": short_trace(out)})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
    messages.append({"role": "user", "content": FORCE_PROMPT.format(style=style)})
    msg = llm.chat(messages)
    trace.append({"forced_answer": True})
    return msg["content"], contexts, trace
