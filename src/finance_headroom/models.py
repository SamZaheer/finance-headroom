"""Thin wrappers over the two model APIs. Same call shape for both.

Requires env vars: ANTHROPIC_API_KEY, OPENAI_API_KEY.
"""
import json
import os

from dotenv import load_dotenv

# must run before the constants below read the environment; never overrides variables that are
# already set (e.g. by docker --env-file)
load_dotenv()

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
OPUS_MODEL = os.environ.get("OPUS_MODEL", "claude-opus-5")
GPT_MODEL = os.environ.get("GPT_MODEL", "gpt-5.1")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")  # set to https://openrouter.ai/api/v1 if OPENAI_API_KEY is an OpenRouter key
MAX_TOOL_TURNS = 10
# the SDKs back off and retry 429 / 5xx / overloaded themselves; raised from their default of 2
# so a large parallel run rides out rate limiting instead of logging failures
MAX_RETRIES = int(os.environ.get("FH_MAX_RETRIES", "8"))
REQUEST_TIMEOUT = float(os.environ.get("FH_REQUEST_TIMEOUT", "300"))


def claude_params(model: str, system_prompt: str, messages: list, claude_tools=None) -> dict:
    """One Messages API request. Shared by the real-time loop below and the Batch API path
    (batch.py), so a batched no-tool run sends exactly what a real-time one would."""
    # Note: the current Anthropic Messages API for this model generation has no
    # temperature/top_p/top_k parameter -- sampling is fixed platform-side, not
    # user-controlled. GPT-5.1 below still exposes temperature=0.
    return {"model": model, "max_tokens": 4096, "system": system_prompt, "messages": messages,
            "tools": claude_tools or []}


def call_claude(system_prompt: str, user_prompt: str, tools=None, tool_executor=None, model=None):
    """Returns (final_answer_text, transcript: list[dict])."""
    import anthropic

    client = anthropic.Anthropic(max_retries=MAX_RETRIES, timeout=REQUEST_TIMEOUT)
    messages = [{"role": "user", "content": user_prompt}]
    transcript = [{"role": "user", "content": user_prompt}]
    claude_tools = None
    if tools:
        claude_tools = [
            {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
            for t in tools
        ]

    for _ in range(MAX_TOOL_TURNS):
        resp = client.messages.create(**claude_params(model or CLAUDE_MODEL, system_prompt, messages, claude_tools))
        transcript.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            final_text = "".join(b.text for b in resp.content if b.type == "text")
            return final_text, transcript

        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            result = tool_executor(block.name, block.input)
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(result)})
            transcript.append({"role": "tool_result", "tool": block.name, "input": block.input, "output": str(result)})
        messages.append({"role": "user", "content": tool_results})

    return "[NO ANSWER: exceeded max tool turns]", transcript


def call_gpt(system_prompt: str, user_prompt: str, tools=None, tool_executor=None):
    """Returns (final_answer_text, transcript: list[dict])."""
    from openai import OpenAI

    client = OpenAI(base_url=OPENAI_BASE_URL, max_retries=MAX_RETRIES, timeout=REQUEST_TIMEOUT)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    transcript = list(messages)
    gpt_tools = None
    if tools:
        gpt_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

    for _ in range(MAX_TOOL_TURNS):
        resp = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            tools=gpt_tools or None,
            temperature=0,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        transcript.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return msg.content or "", transcript

        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            result = tool_executor(call.function.name, args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
            transcript.append({"role": "tool", "tool_call_id": call.id, "name": call.function.name, "input": args, "output": str(result)})

    return "[NO ANSWER: exceeded max tool turns]", transcript


def call_opus(system_prompt: str, user_prompt: str, tools=None, tool_executor=None):
    """Same as call_claude, pinned to the Opus model. Added per client request (2026-09)."""
    return call_claude(system_prompt, user_prompt, tools=tools, tool_executor=tool_executor, model=OPUS_MODEL)


CALLERS = {"claude": call_claude, "opus": call_opus, "gpt": call_gpt}
ANTHROPIC_MODEL_IDS = {"claude": CLAUDE_MODEL, "opus": OPUS_MODEL}  # models the Batch API path can serve
