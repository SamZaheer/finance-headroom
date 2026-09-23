SYSTEM_PROMPT = (
    "You are a financial analyst. Cite the specific source excerpt that supports each "
    "factual claim in your answer, using [DOC] tags. If the periods, entities, or "
    "accounting bases being compared are not directly comparable (e.g. a divestiture, "
    "spinoff, segment redefinition, or fiscal-year mismatch), say so explicitly before "
    "giving a final numeric answer. Give your final answer on its own line as: "
    "ANSWER: <value or text>"
)

NO_TOOL_TEMPLATE = """{excerpts_block}

Question: {question}"""


def excerpts_block(item_id: str, corpus_dir) -> str:
    docs = []
    for i, f in enumerate(sorted((corpus_dir / item_id).glob("*.txt")), start=1):
        docs.append(f"[DOC-{i} | {f.name}]\n{f.read_text().strip()}")
    return "<excerpts>\n" + "\n\n".join(docs) + "\n</excerpts>"


TOOL_TEMPLATE = """You have access to search_corpus (searches only this question's source
filings) and python_eval (a calculator). You may make up to 10 tool calls.

Question: {question}"""
