"""Anthropic Message Batches for the no-tool condition: about half the per-token price, results
within 24h instead of immediately. Only no-tool runs qualify (one request each); tool runs are
multi-turn, so each turn would need its own batch round trip.

Used through fh-run:
    fh-run --batch             # claude/opus no-tool jobs -> submitted as batches, everything else
                               # runs in real time as usual; then waits and collects the batches
    fh-run --batch --no-wait   # submit and exit; collect later with:
    fh-run --collect           # write transcripts for every finished batch, report the rest

Each submitted batch is recorded in results/batches/<batch_id>.json before anything else
happens, so an interrupted run never re-submits or loses a batch: jobs in a batch that
hasn't been collected yet are skipped by the next fh-run.
GPT stays real-time: its OpenRouter batch variant does not serve the chat-completions API.
"""
import json
import time
from datetime import UTC, datetime

from .models import ANTHROPIC_MODEL_IDS, MAX_RETRIES, REQUEST_TIMEOUT, claude_params
from .paths import RESULTS
from .prompts import SYSTEM_PROMPT

BATCH_DIR = RESULTS / "batches"
MAX_REQUESTS_PER_BATCH = 10_000  # API limit is 100k / 256MB; smaller batches finish and fail independently


def eligible(job) -> bool:
    model_key, tool_condition, _, _ = job
    return tool_condition == "no_tool" and model_key in ANTHROPIC_MODEL_IDS


def custom_id(model_key, tool_condition, item_id, repeat) -> str:
    return f"{model_key}__{tool_condition}__{item_id}__r{repeat}"  # [A-Za-z0-9_-]{1,64}


def parse_custom_id(cid: str):
    model_key, tool_condition, item_id, r = cid.split("__")
    return model_key, tool_condition, item_id, int(r[1:])


def default_client():
    import anthropic
    return anthropic.Anthropic(max_retries=MAX_RETRIES, timeout=REQUEST_TIMEOUT)


def _records():
    return sorted(BATCH_DIR.glob("*.json")) if BATCH_DIR.exists() else []


def in_flight() -> set:
    """(model, condition, item_id, repeat) of every job in a batch not yet collected."""
    keys = set()
    for p in _records():
        rec = json.loads(p.read_text())
        if rec["status"] != "collected":
            keys.update(parse_custom_id(c) for c in rec["custom_ids"])
    return keys


def submit(jobs, user_prompt, write_atomic, client=None) -> list:
    """Submit eligible jobs; returns batch ids. Each batch is recorded on disk before returning."""
    client = client or default_client()
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    ids = []
    for start in range(0, len(jobs), MAX_REQUESTS_PER_BATCH):
        chunk = jobs[start:start + MAX_REQUESTS_PER_BATCH]
        requests = [{
            "custom_id": custom_id(m, c, item["id"], r),
            "params": claude_params(ANTHROPIC_MODEL_IDS[m], SYSTEM_PROMPT,
                                    [{"role": "user", "content": user_prompt(c, item)}]),
        } for m, c, item, r in chunk]
        batch = client.messages.batches.create(requests=requests)
        write_atomic(BATCH_DIR / f"{batch.id}.json", json.dumps({
            "batch_id": batch.id, "status": "submitted",
            "submitted_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "custom_ids": [r["custom_id"] for r in requests],
        }, indent=2))
        ids.append(batch.id)
        print(f"submitted batch {batch.id} with {len(requests)} no-tool requests")
    return ids


def collect(items_by_id, user_prompt, save_transcript, write_atomic, client=None):
    """Write transcripts for every ended, uncollected batch. Returns (written, failures, still_running)."""
    client = client or default_client()
    written, failures, running = 0, [], 0
    for p in _records():
        rec = json.loads(p.read_text())
        if rec["status"] == "collected":
            continue
        batch = client.messages.batches.retrieve(rec["batch_id"])
        if batch.processing_status != "ended":
            running += 1
            c = batch.request_counts
            print(f"batch {rec['batch_id']}: {batch.processing_status} "
                  f"({c.succeeded} succeeded, {c.processing} processing, {c.errored} errored)")
            continue
        for res in client.messages.batches.results(rec["batch_id"]):
            m, c, item_id, r = parse_custom_id(res.custom_id)
            if res.result.type == "succeeded":
                msg = res.result.message
                transcript = [{"role": "user", "content": user_prompt(c, items_by_id[item_id])},
                              {"role": "assistant", "content": [b.model_dump() for b in msg.content]}]
                final_text = "".join(b.text for b in msg.content if b.type == "text")
                save_transcript(m, c, item_id, r, final_text, transcript)
                written += 1
            else:  # errored / canceled / expired -> not written, so the next fh-run retries it
                failures.append({"model": m, "tool_condition": c, "item_id": item_id, "repeat": r,
                                 "error": f"batch {res.result.type}: {getattr(res.result, 'error', '')}"})
        rec.update(status="collected", collected_at=datetime.now(UTC).isoformat(timespec="seconds"))
        write_atomic(p, json.dumps(rec, indent=2))
        print(f"collected batch {rec['batch_id']}")
    return written, failures, running


def wait_and_collect(items_by_id, user_prompt, save_transcript, write_atomic, poll_seconds=60, client=None):
    client = client or default_client()
    written, failures = 0, []
    while True:
        w, f, running = collect(items_by_id, user_prompt, save_transcript, write_atomic, client)
        written += w
        failures += f
        if not running:
            return written, failures
        time.sleep(poll_seconds)
