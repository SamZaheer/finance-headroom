"""Deterministic scoring pass. Fills numeric/final-answer correctness automatically and
leaves the rubric columns (comparability_flag / evidence_grounding / tool_use_correctness /
failure_category / evidence_quote) for judgment items to the grading pass: reviewed drafts
for repeat 1, the calibrated judge (fh-judge) for later repeats. Existing grades are
preserved across re-runs.
"""
import csv
import json
import re

from .paths import ANSWER_KEYS, TRANSCRIPTS
from .paths import SCORED as OUT

MANUAL_COLUMNS = ["comparability_flag", "evidence_grounding", "tool_use_correctness", "failure_category", "evidence_quote"]


def load_answer_keys():
    keys = (json.loads(line) for line in ANSWER_KEYS.read_text().splitlines() if line.strip())
    return {k["id"]: k for k in keys}


SCALE_TO_MILLIONS = {"thousand": 0.001, "million": 1, "billion": 1_000, "trillion": 1_000_000}


def extract_answer_value(text: str):
    m = re.search(r"ANSWER:\s*(.+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


def numeric_match(model_answer: str, expected: dict) -> bool:
    value_range = expected.get("value_range") or [None, None]
    lo, hi = value_range
    if lo is None or hi is None:
        return None  # not a numeric item (judgment/justification), needs manual grading
    found = re.findall(r"(-?\d+\.?\d*)\s*(thousand|million|billion|trillion)?", model_answer.replace(",", ""), re.IGNORECASE)
    if not found:
        return False
    tol = expected.get("tolerance", 0)
    # for dollar-millions answers, honor an explicitly stated scale ("$637.959 billion",
    # "$768,044 thousand") -- a bare number is still taken as-is, so an unlabeled
    # thousands figure reported as millions stays wrong
    in_millions = expected.get("unit") == "USD millions"
    values = [float(n) * (SCALE_TO_MILLIONS[s.lower()] if in_millions and s else 1) for n, s in found]
    # models often state an intermediate value before the final one (e.g. "from 7.80% to
    # 8.02%, an increase of 0.22pp") -- credit a match if ANY stated number lands in range,
    # not just the first one found.
    return any((lo - tol) <= v <= (hi + tol) for v in values)


def load_existing_manual_grades():
    """(model, tool_condition, item_id, repeat) -> {manual column: value}, so re-running this
    script after new items/transcripts show up doesn't wipe out a prior grading pass.
    Rows written before repeats existed have no repeat column and count as repeat 1."""
    if not OUT.exists():
        return {}
    existing = {}
    for row in csv.DictReader(OUT.open()):
        key = (row["model"], row["tool_condition"], row["item_id"], row.get("repeat") or "1")
        existing[key] = {c: row.get(c, "") for c in MANUAL_COLUMNS}
    return existing


def main():
    keys = load_answer_keys()
    prior_manual_grades = load_existing_manual_grades()
    rows = []
    for path in sorted(TRANSCRIPTS.glob("*.json")):
        run = json.loads(path.read_text())
        item_id = run["item_id"]
        key = keys.get(item_id, {})
        model_answer = extract_answer_value(run["final_answer"])
        expected = key.get("expected_final_answer", {})
        correct = numeric_match(model_answer, expected)
        row = {
            "model": run["model"],
            "tool_condition": run["tool_condition"],
            "item_id": item_id,
            "repeat": str(run.get("repeat", 1)),
            "bucket": key.get("bucket", ""),
            "mechanism": key.get("mechanism", ""),
            "model_answer": model_answer,
            "expected": json.dumps(expected),
            "auto_correct": correct,
        }
        manual = prior_manual_grades.get((run["model"], run["tool_condition"], item_id, row["repeat"]), {})
        row.update({c: manual.get(c, "") for c in MANUAL_COLUMNS})
        rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
