"""The two tools models get in the "realistic-tool" condition.

search_corpus: keyword search over the frozen per-item filing excerpts only, not the open
  web, so every run sees the same evidence and the answer key stays reproducible.
python_eval: a calculator, so arithmetic errors are separable from reasoning errors. It is a
  python3 subprocess with a 5 s timeout, not a security sandbox; isolation comes from the
  container it runs in (see README, "Isolation and reward hacking").
"""
import re
import subprocess

from .paths import CORPUS as CORPUS_DIR

TOOL_SPECS = [
    {
        "name": "search_corpus",
        "description": (
            "Search the source filing excerpts provided for this question. "
            "Returns the most relevant passages. Does not search the open web."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search keywords"}},
            "required": ["query"],
        },
    },
    {
        "name": "python_eval",
        "description": "Run a short Python expression/snippet for arithmetic. Print the result.",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code; print() the answer."}},
            "required": ["code"],
        },
    },
]


def _paragraphs(item_id: str):
    item_dir = CORPUS_DIR / item_id
    paras = []
    for f in sorted(item_dir.glob("*.txt")):
        for p in f.read_text().split("\n\n"):
            p = p.strip()
            if p:
                paras.append((f.name, p))
    return paras


def search_corpus(query: str, item_id: str, top_k: int = 3) -> str:
    terms = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
    scored = []
    for doc, para in _paragraphs(item_id):
        text = para.lower()
        score = sum(text.count(t) for t in terms)
        if score:
            scored.append((score, doc, para))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return "No matching passages found."
    return "\n\n".join(f"[{doc}] {para}" for _, doc, para in scored[:top_k])


def python_eval(code: str) -> str:
    try:
        out = subprocess.run(
            ["python3", "-c", code], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or out.stderr.strip() or "(no output -- did you print()?)"
    except subprocess.TimeoutExpired:
        return "Error: execution timed out"


def make_executor(item_id: str):
    def executor(tool_name: str, tool_input: dict) -> str:
        if tool_name == "search_corpus":
            return search_corpus(tool_input["query"], item_id)
        if tool_name == "python_eval":
            return python_eval(tool_input["code"])
        return f"Unknown tool: {tool_name}"

    return executor
