"""Runs the model x tool-condition x item (x repeat) matrix in parallel.

Usage:
    fh-run                                   # everything, 1 repeat, resume-safe
    fh-run --repeats 3 --workers 8           # 3 samples per run, 8 concurrent calls per provider
    fh-run --models opus,gpt --conditions tool --items AIR- --limit 20
    fh-run --batch [--no-wait]               # claude/opus no-tool jobs via the Batch API (see batch.py)
    fh-run --collect                         # collect finished batches
Requires: ANTHROPIC_API_KEY, OPENAI_API_KEY in env.

API calls are network-bound, so a thread pool is enough; a per-provider semaphore keeps each
provider under its rate limit, and the SDKs retry 429/5xx themselves (models.MAX_RETRIES).
"""
import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

from . import models
from .models import CALLERS
from .paths import CORPUS, QUESTIONS, RESULTS, TRANSCRIPTS
from .prompts import NO_TOOL_TEMPLATE, SYSTEM_PROMPT, TOOL_TEMPLATE, excerpts_block
from .tools import TOOL_SPECS, make_executor

MODELS = ["claude", "opus", "gpt"]
TOOL_CONDITIONS = ["no_tool", "tool"]
PROVIDER = {"claude": "anthropic", "opus": "anthropic", "gpt": "openai"}


def load_items():
    return [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]


def transcript_path(model_key: str, tool_condition: str, item_id: str, repeat: int = 1):
    # repeat 1 keeps the original name, so transcripts from before --repeats existed still count
    suffix = "" if repeat == 1 else f"_r{repeat}"
    return TRANSCRIPTS / f"{model_key}_{tool_condition}_{item_id}{suffix}.json"


def write_atomic(path, text: str):
    # write-then-rename: a crash mid-write can never leave a half-written file that
    # the resume check would then treat as done
    path.parent.mkdir(parents=True, exist_ok=True)  # a missing folder is created, never an error
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def select(items, model_arg, condition_arg, item_arg, repeats):
    """Every (model, condition, item, repeat) job requested, as a list of tuples."""
    models_ = model_arg.split(",") if model_arg else MODELS
    conditions = condition_arg.split(",") if condition_arg else TOOL_CONDITIONS
    unknown = set(models_) - set(MODELS) | set(conditions) - set(TOOL_CONDITIONS)
    if unknown:
        raise SystemExit(f"unknown model/condition: {sorted(unknown)}")
    prefixes = item_arg.split(",") if item_arg else [""]
    chosen = [it for it in items if any(it["id"].startswith(p) for p in prefixes)]
    return [(m, c, it, r) for m in models_ for c in conditions for it in chosen for r in range(1, repeats + 1)]


def user_prompt(tool_condition: str, item: dict) -> str:
    if tool_condition == "no_tool":
        return NO_TOOL_TEMPLATE.format(excerpts_block=excerpts_block(item["id"], CORPUS), question=item["question"])
    return TOOL_TEMPLATE.format(question=item["question"])


def save_transcript(model_key, tool_condition, item_id, repeat, final_text, transcript):
    out_path = transcript_path(model_key, tool_condition, item_id, repeat)
    write_atomic(out_path, json.dumps({
        "model": model_key,
        "tool_condition": tool_condition,
        "item_id": item_id,
        "repeat": repeat,
        "final_answer": final_text,
        "transcript": transcript,
    }, indent=2, default=str))
    return out_path.name


def run_one(model_key: str, tool_condition: str, item: dict, repeat: int = 1):
    caller = CALLERS[model_key]
    prompt = user_prompt(tool_condition, item)
    if tool_condition == "no_tool":
        final_text, transcript = caller(SYSTEM_PROMPT, prompt, tools=None, tool_executor=None)
    else:
        final_text, transcript = caller(SYSTEM_PROMPT, prompt, tools=TOOL_SPECS, tool_executor=make_executor(item["id"]))
    return save_transcript(model_key, tool_condition, item["id"], repeat, final_text, transcript)


def prompt_hash() -> str:
    blob = json.dumps([SYSTEM_PROMPT, NO_TOOL_TEMPLATE, TOOL_TEMPLATE, TOOL_SPECS], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fh-run", description=__doc__.split("\n")[0])
    ap.add_argument("--models", help=f"comma-separated subset of {','.join(MODELS)}")
    ap.add_argument("--conditions", help=f"comma-separated subset of {','.join(TOOL_CONDITIONS)}")
    ap.add_argument("--items", help="comma-separated item-id prefixes, e.g. AIR-,TECH-CTRL-01")
    ap.add_argument("--repeats", type=int, default=1, help="samples per (model, condition, item)")
    ap.add_argument("--workers", type=int, default=4, help="concurrent API calls per provider")
    ap.add_argument("--limit", type=int, help="run at most N pending jobs (for a cheap trial)")
    ap.add_argument("--batch", action="store_true", help="send claude/opus no-tool jobs through the Batch API")
    ap.add_argument("--no-wait", action="store_true", help="with --batch: submit and exit, collect later")
    ap.add_argument("--collect", action="store_true", help="collect finished batches and exit")
    ap.add_argument("--poll-seconds", type=int, default=60, help="batch status poll interval")
    args = ap.parse_args(argv)
    from . import batch as batches  # local import: batch.py needs nothing from here at import time

    TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    items_by_id = {it["id"]: it for it in load_items()}
    if args.collect:
        written, failures, running = batches.collect(items_by_id, user_prompt, save_transcript, write_atomic)
        write_atomic(RESULTS / "run_failures.jsonl", "".join(json.dumps(f) + "\n" for f in failures))
        print(f"collected {written} transcripts, {len(failures)} failed, {running} batch(es) still running")
        return
    jobs = select(list(items_by_id.values()), args.models, args.conditions, args.items, args.repeats)
    waiting = batches.in_flight()  # submitted earlier, not collected yet: never submit twice
    n_done = n_waiting = 0
    pending = []
    for j in jobs:
        if transcript_path(j[0], j[1], j[2]["id"], j[3]).exists():
            n_done += 1
        elif (j[0], j[1], j[2]["id"], j[3]) in waiting:
            n_waiting += 1
        else:
            pending.append(j)
    if args.limit is not None:
        pending = pending[: args.limit]
    to_batch, now = [], []
    for j in pending:
        (to_batch if args.batch and batches.eligible(j) else now).append(j)
    pending = now
    print(f"{len(jobs)} requested: {n_done} already done, {n_waiting} in submitted batches, "
          f"{len(to_batch)} to batch, {len(pending)} to run now")
    batch_ids = batches.submit(to_batch, user_prompt, write_atomic) if to_batch else []

    gates = {p: threading.Semaphore(args.workers) for p in set(PROVIDER.values())}
    lock = threading.Lock()
    failures, done = [], 0
    started = time.monotonic()

    def job(m, c, item, r):
        with gates[PROVIDER[m]]:
            return run_one(m, c, item, r)

    with ThreadPoolExecutor(max_workers=args.workers * len(gates)) as pool:
        futures = {pool.submit(job, *j): j for j in pending}
        for fut in as_completed(futures):
            m, c, item, r = futures[fut]
            with lock:
                try:
                    fut.result()
                    done += 1
                    status = "done"
                except Exception as e:  # one bad run must not stop the batch
                    failures.append({"model": m, "tool_condition": c, "item_id": item["id"], "repeat": r,
                                     "error": f"{type(e).__name__}: {e}"})
                    status = f"FAILED ({type(e).__name__})"
                n = done + len(failures)
                print(f"[{n}/{len(pending)}] {status}: {m} {c} {item['id']} r{r}  ({time.monotonic() - started:.0f}s)")

    if batch_ids and not args.no_wait:
        print(f"waiting for {len(batch_ids)} batch(es); safe to interrupt -- resume with fh-run --collect")
        written, batch_failures = batches.wait_and_collect(items_by_id, user_prompt, save_transcript, write_atomic,
                                                           args.poll_seconds)
        done += written
        failures += batch_failures
    elif batch_ids:
        print(f"submitted {len(batch_ids)} batch(es); collect later with fh-run --collect")

    write_atomic(RESULTS / "run_failures.jsonl", "".join(json.dumps(f) + "\n" for f in failures))
    write_atomic(RESULTS / "run_manifest.json", json.dumps({
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_ids": {"claude": models.CLAUDE_MODEL, "opus": models.OPUS_MODEL, "gpt": models.GPT_MODEL},
        "prompt_hash": prompt_hash(),
        "args": vars(args),
        "requested": len(jobs), "ran": len(pending), "batched": len(to_batch), "batch_ids": batch_ids,
        "succeeded": done, "failed": len(failures),
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }, indent=2))
    print(f"finished: {done} succeeded, {len(failures)} failed"
          + (" -- see results/run_failures.jsonl, re-run fh-run to retry them" if failures else ""))


if __name__ == "__main__":
    main()
