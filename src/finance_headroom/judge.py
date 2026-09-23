"""Automated judgment grading (fh-judge): an LLM judge from a different model family than the
graded models scores judgment rows against each item's pre-written rubric.

Workflow:
    fh-judge --calibrate      # judge the rows a human already graded; must reach >= 90% agreement
    fh-judge                  # judge every ungraded judgment row in results/scored.csv
    fh-judge --apply          # write high-confidence judge grades into scored.csv; low-confidence
                              # rows + a random 5% spot-check go to grading_draft.csv for a human
    fh-judge --merge-approved # merge grading_draft.csv rows marked approved=yes into scored.csv

--apply refuses to run unless the last calibration of the same judge model + prompt passed.
The judge is reached through any OpenAI-compatible endpoint (OpenRouter by default):
    FH_JUDGE_MODEL (default google/gemini-3.8-flash), FH_JUDGE_BASE_URL (default OPENAI_BASE_URL),
    FH_JUDGE_API_KEY (default OPENAI_API_KEY).
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from . import models  # noqa: F401  (loads .env before the settings below are read)
from .paths import ANSWER_KEYS, RESULTS, SCORED, TRANSCRIPTS

JUDGE_MODEL = os.environ.get("FH_JUDGE_MODEL", "google/gemini-3.8-flash")
JUDGE_BASE_URL = os.environ.get("FH_JUDGE_BASE_URL", os.environ.get("OPENAI_BASE_URL"))
JUDGE_API_KEY = os.environ.get("FH_JUDGE_API_KEY", os.environ.get("OPENAI_API_KEY"))
AGREEMENT_THRESHOLD = 0.90
SPOT_CHECK_RATE = 0.05
JUDGE_ATTEMPTS = 3

GRADE_COLUMNS = ["comparability_flag", "evidence_grounding", "tool_use_correctness", "failure_category", "evidence_quote"]
FLAGS = ["correct", "partial", "incorrect"]
GROUNDING = ["cites_relevant_note", "uncited", "misread_evidence"]
TOOL_USE = ["n/a", "effective", "incomplete", "ineffective"]
FAILURES = ["unsupported_conclusion", "over_hedging", "wrong_entity", "wrong_period_basis", "computation",
            "unit_scale", "cross_document_reconciliation", "definition_accounting_error"]

CALIBRATION_JSON = RESULTS / "judge_calibration.json"
CALIBRATION_CSV = RESULTS / "judge_calibration.csv"
JUDGE_CSV = RESULTS / "judge_grades.csv"
DRAFT_CSV = RESULTS / "grading_draft.csv"
CACHE = RESULTS / "judge_cache.jsonl"

SYSTEM = """You grade answers from a finance-AI evaluation. Apply ONLY the item's rubric and answer key.
Grade the substance of the final answer, not its style. Rules:
- "correct": reaches the answer the key requires, with supporting figures consistent with the key.
- "partial": right direction but misses what the rubric requires (e.g. names one winner without
  acknowledging the other dimension's leader, flags an issue but does not adjust for it, a minor
  error below the top of a ranking).
- "incorrect": wrong conclusion, or uses a figure the key says is not comparable without flagging it.
- If the item has a "grading_rubric", it overrides these general rules.
- evidence_quote must be copied EXACTLY from the final answer (under 30 words), no paraphrase.
- confidence "low" when a reasonable grader could disagree; say why in rationale.
- tool_use_correctness is "n/a" when the run had no tools; otherwise judge from the tool log whether
  the model retrieved what it needed.
Return only the JSON object."""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["comparability_flag", "evidence_grounding", "tool_use_correctness", "failure_category",
                 "evidence_quote", "confidence", "rationale"],
    "properties": {
        "comparability_flag": {"type": "string", "enum": FLAGS},
        "evidence_grounding": {"type": "string", "enum": GROUNDING},
        "tool_use_correctness": {"type": "string", "enum": TOOL_USE},
        "failure_category": {"type": "array", "items": {"type": "string", "enum": FAILURES}},
        "evidence_quote": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "low"]},
        "rationale": {"type": "string"},
    },
}


def prompt_hash() -> str:
    return hashlib.sha256(json.dumps([SYSTEM, SCHEMA], sort_keys=True).encode()).hexdigest()[:16]


def load_keys():
    keys = (json.loads(line) for line in ANSWER_KEYS.read_text().splitlines() if line.strip())
    return {k["id"]: k for k in keys}


def transcript_file(model, tool_condition, item_id, repeat="1"):
    suffix = "" if str(repeat) == "1" else f"_r{repeat}"
    return TRANSCRIPTS / f"{model}_{tool_condition}_{item_id}{suffix}.json"


def tool_log(transcript):
    """The tool calls in a transcript (both SDK shapes) as [{tool, input, output}]."""
    return [{"tool": t.get("tool") or t.get("name"), "input": t.get("input"), "output": str(t["output"])}
            for t in transcript if isinstance(t, dict) and (t.get("tool") or t.get("name")) and "output" in t]


def compact(calls, limit=600):
    # every caller (fh-judge, env replay) goes through this, so identical runs hit the same cache key
    return [{"tool": c["tool"], "input": c["input"],
             "output": c["output"][:limit] + ("..." if len(c["output"]) > limit else "")} for c in calls]


def build_prompt(key: dict, final_answer: str, tools_used: bool, calls: list) -> str:
    item = {k: key.get(k) for k in ("id", "question", "relevant_evidence", "expected_intermediate_steps",
                                     "expected_final_answer", "trap", "grading_rubric") if key.get(k) is not None}
    return (f"ITEM AND ANSWER KEY:\n{json.dumps(item, indent=2)}\n\n"
            f"RUN HAD TOOLS: {'yes' if tools_used else 'no'}\n"
            + (f"TOOL LOG:\n{json.dumps(calls, indent=1)}\n\n" if tools_used else "\n")
            + f"FINAL ANSWER TO GRADE:\n<<<\n{final_answer}\n>>>")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("’", "'").replace("“", '"').replace("”", '"')).strip().lower()


def validate(grade: dict, final_answer: str, tools_used: bool) -> dict:
    """Coerce the judge's JSON into the scored.csv columns; downgrade confidence on anything off."""
    notes = []
    g = dict(grade)
    if g.get("comparability_flag") not in FLAGS:
        raise ValueError(f"bad comparability_flag: {g.get('comparability_flag')!r}")
    quote = (g.get("evidence_quote") or "").strip().strip('"')
    if not quote or _norm(quote) not in _norm(final_answer):
        notes.append("quote not found verbatim in answer")
    if not tools_used and g.get("tool_use_correctness") != "n/a":
        g["tool_use_correctness"] = "n/a"
    fails = [f for f in (g.get("failure_category") or []) if f in FAILURES]
    if g["comparability_flag"] == "correct":
        fails = []
    confidence = g.get("confidence") if g.get("confidence") in ("high", "low") else "low"
    if notes:
        confidence = "low"
    rationale = (g.get("rationale") or "").strip()
    return {
        "comparability_flag": g["comparability_flag"],
        "evidence_grounding": g.get("evidence_grounding") if g.get("evidence_grounding") in GROUNDING else "uncited",
        "tool_use_correctness": g.get("tool_use_correctness") if g.get("tool_use_correctness") in TOOL_USE else "n/a",
        "failure_category": ";".join(fails),
        "evidence_quote": f'{rationale} Quote: "{quote}"' + (f" [{'; '.join(notes)}]" if notes else ""),
        "confidence": confidence,
    }


class Judge:
    """Calls the judge model, with a persistent cache keyed on (model, prompt, answer), so
    calibration, grading and env replay never pay twice for the same answer."""

    def __init__(self, model=JUDGE_MODEL, client=None):
        self.model = model
        self._client = client
        self._lock = threading.Lock()
        self._cache = {}
        if CACHE.exists():
            for line in CACHE.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._cache[rec["k"]] = rec["v"]

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY,
                                  max_retries=models.MAX_RETRIES, timeout=models.REQUEST_TIMEOUT)
        return self._client

    def grade(self, key: dict, final_answer: str, tools_used: bool, calls: list) -> dict:
        prompt = build_prompt(key, final_answer, tools_used, compact(calls))
        k = hashlib.sha256(f"{self.model}|{prompt_hash()}|{prompt}".encode()).hexdigest()
        with self._lock:
            if k in self._cache:
                return validate(self._cache[k], final_answer, tools_used)
        for attempt in range(JUDGE_ATTEMPTS):
            # the judge occasionally returns cut-off JSON; a malformed reply is retried, never cached
            resp = self.client.chat.completions.create(
                model=self.model, temperature=0, max_tokens=8000,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                response_format={"type": "json_schema", "json_schema": {"name": "grade", "strict": True, "schema": SCHEMA}},
            )
            try:
                raw = json.loads(resp.choices[0].message.content or "{}")
                result = validate(raw, final_answer, tools_used)
                break
            except ValueError:  # includes json.JSONDecodeError
                if attempt == JUDGE_ATTEMPTS - 1:
                    raise
        with self._lock:
            self._cache[k] = raw
            RESULTS.mkdir(parents=True, exist_ok=True)
            with CACHE.open("a") as f:
                f.write(json.dumps({"k": k, "v": raw}) + "\n")
        return result


def judge_rows(rows, judge: Judge, keys: dict, workers: int):
    """Grades each scored.csv row; returns (row, grade-or-None, error-or-None) in input order."""
    def one(r):
        run = json.loads(transcript_file(r["model"], r["tool_condition"], r["item_id"], r.get("repeat", "1")).read_text())
        tools_used = r["tool_condition"] == "tool"
        return judge.grade(keys[r["item_id"]], run["final_answer"], tools_used, tool_log(run["transcript"]) if tools_used else [])

    def safe(r):
        try:
            return r, one(r), None
        except Exception as e:  # one bad judge call must not stop the batch
            return r, None, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(safe, rows))


def row_key(r):
    return (r["model"], r["tool_condition"], r["item_id"], r.get("repeat") or "1")


def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def calibrate(scored, judge, keys, workers):
    human = [r for r in scored if r["auto_correct"] == "" and r["comparability_flag"]]
    if not human:
        raise SystemExit("no human-graded judgment rows to calibrate against")
    results = judge_rows(human, judge, keys, workers)
    out, agree, errors = [], 0, 0
    confusion = collections.Counter()
    for r, g, err in results:
        if err:
            errors += 1
            out.append({**{k: r[k] for k in ("model", "tool_condition", "item_id", "bucket")}, "human": r["comparability_flag"],
                        "judge": "", "match": "", "error": err})
            continue
        match = g["comparability_flag"] == r["comparability_flag"]
        agree += match
        confusion[(r["comparability_flag"], g["comparability_flag"])] += 1
        out.append({**{k: r[k] for k in ("model", "tool_condition", "item_id", "bucket")}, "human": r["comparability_flag"],
                    "judge": g["comparability_flag"], "match": match, "confidence": g["confidence"],
                    "judge_note": g["evidence_quote"], "error": ""})
    graded = len(human) - errors
    rate = agree / graded if graded else 0.0
    passed = rate >= AGREEMENT_THRESHOLD and errors == 0
    write_csv(CALIBRATION_CSV, out, ["model", "tool_condition", "item_id", "bucket", "human", "judge", "match",
                                     "confidence", "judge_note", "error"])
    CALIBRATION_JSON.write_text(json.dumps({
        "judge_model": judge.model, "prompt_hash": prompt_hash(), "rows": len(human), "graded": graded,
        "errors": errors, "agreement": round(rate, 4), "threshold": AGREEMENT_THRESHOLD, "passed": passed,
        "confusion_human_to_judge": {f"{h}->{j}": n for (h, j), n in sorted(confusion.items())},
    }, indent=2))
    print(f"calibration: judge {judge.model} agrees with human grades on {agree}/{graded} = {rate:.1%}"
          f" ({errors} errors) -> {'PASS' if passed else 'FAIL'} (threshold {AGREEMENT_THRESHOLD:.0%})")
    for (h, j), n in sorted(confusion.items()):
        if h != j:
            print(f"  human {h:9} -> judge {j:9}: {n}")
    print(f"per-row detail: {CALIBRATION_CSV}")
    return passed


def calibration_ok(model):
    if not CALIBRATION_JSON.exists():
        return False, "no calibration found -- run fh-judge --calibrate first"
    c = json.loads(CALIBRATION_JSON.read_text())
    if c["judge_model"] != model or c["prompt_hash"] != prompt_hash():
        return False, "last calibration used a different judge model or prompt -- recalibrate"
    if not c["passed"]:
        return False, f"last calibration failed ({c['agreement']:.1%} < {c['threshold']:.0%})"
    return True, f"calibration passed at {c['agreement']:.1%}"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fh-judge", description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--calibrate", action="store_true", help="judge human-graded rows and report agreement")
    mode.add_argument("--apply", action="store_true", help="write judge grades into scored.csv (needs a passed calibration)")
    mode.add_argument("--merge-approved", action="store_true", help="merge grading_draft.csv rows with approved=yes")
    ap.add_argument("--model", default=JUDGE_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=0, help="spot-check sampling seed")
    args = ap.parse_args(argv)

    scored = list(csv.DictReader(SCORED.open()))
    fields = list(scored[0].keys())
    keys = load_keys()

    if args.merge_approved:
        drafts = {row_key(d): d for d in csv.DictReader(DRAFT_CSV.open())
                  if d.get("approved", "").strip().lower() in ("yes", "y", "true", "1")}
        n = 0
        for r in scored:
            if row_key(r) in drafts:
                r.update({c: drafts[row_key(r)][c] for c in GRADE_COLUMNS})
                n += 1
        write_csv(SCORED, scored, fields)
        print(f"merged {n} approved draft grades into {SCORED}")
        return

    judge = Judge(args.model)
    if args.calibrate:
        raise SystemExit(0 if calibrate(scored, judge, keys, args.workers) else 1)

    if args.apply:
        ok, msg = calibration_ok(args.model)
        if not ok:
            raise SystemExit(f"refusing to apply: {msg}")
        print(msg)
        grades = {row_key(g): g for g in csv.DictReader(JUDGE_CSV.open())} if JUDGE_CSV.exists() else {}
        # rows already queued for human review stay queued: re-running --apply must never re-sample
        # the spot-check and auto-apply a row a human was supposed to see
        queued = {row_key(d) for d in csv.DictReader(DRAFT_CSV.open())
                  if d.get("grader", "").startswith("judge:") and not d.get("approved", "").strip()} if DRAFT_CSV.exists() else set()
        pending = [r for r in scored if r["auto_correct"] == "" and not r["comparability_flag"]
                   and row_key(r) in grades and row_key(r) not in queued]
        rng = random.Random(args.seed)
        spot = {row_key(r) for r in pending if rng.random() < SPOT_CHECK_RATE}
        to_review, applied = [], 0
        for r in pending:
            g = grades[row_key(r)]
            if g["confidence"] == "low" or row_key(r) in spot:
                to_review.append({**{k: r[k] for k in ("model", "tool_condition", "item_id", "repeat", "bucket")},
                                  **{c: g[c] for c in GRADE_COLUMNS}, "confidence": g["confidence"],
                                  "review_reason": "low confidence" if g["confidence"] == "low" else "random spot-check",
                                  "grader": f"judge:{args.model}", "approved": ""})
            else:
                r.update({c: g[c] for c in GRADE_COLUMNS})
                applied += 1
        write_csv(SCORED, scored, fields)
        existing = list(csv.DictReader(DRAFT_CSV.open())) if DRAFT_CSV.exists() else []
        seen = {row_key(d) for d in to_review}
        draft_fields = ["model", "tool_condition", "item_id", "repeat", "bucket", *GRADE_COLUMNS, "confidence",
                        "review_reason", "grader", "approved"]
        write_csv(DRAFT_CSV, [d for d in existing if row_key(d) not in seen] + to_review, draft_fields)
        print(f"applied {applied} judge grades to {SCORED}; {len(to_review)} rows sent to {DRAFT_CSV} for review "
              f"({sum(d['review_reason'] == 'low confidence' for d in to_review)} low-confidence, "
              f"{sum(d['review_reason'] == 'random spot-check' for d in to_review)} spot-check)")
        return

    pending = [r for r in scored if r["auto_correct"] == "" and not r["comparability_flag"]]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"judging {len(pending)} ungraded judgment rows with {args.model}")
    out, errors = [], 0
    for r, g, err in judge_rows(pending, judge, keys, args.workers):
        if err:
            errors += 1
            print(f"  ERROR {r['model']} {r['tool_condition']} {r['item_id']} r{r.get('repeat', '1')}: {err}")
            continue
        out.append({**{k: r[k] for k in ("model", "tool_condition", "item_id", "bucket")},
                    "repeat": r.get("repeat") or "1", **g, "judge_model": args.model, "prompt_hash": prompt_hash()})
    prior = {row_key(g): g for g in csv.DictReader(JUDGE_CSV.open())} if JUDGE_CSV.exists() else {}
    prior.update({row_key(g): g for g in out})
    write_csv(JUDGE_CSV, list(prior.values()), ["model", "tool_condition", "item_id", "repeat", "bucket", *GRADE_COLUMNS,
                                               "confidence", "judge_model", "prompt_hash"])
    print(f"wrote {len(out)} judge grades to {JUDGE_CSV} ({errors} errors); next: fh-judge --apply")


if __name__ == "__main__":
    main()
