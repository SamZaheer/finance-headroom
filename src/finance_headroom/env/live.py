"""Live Gymnasium episodes (fh-gym): the env drives real LLM calls -- no transcript replay.

Each episode: env.reset(item) -> the model gets the question -> every tool call the model makes is
executed BY THE ENV via env.step({"tool": ...}) (step cost applies) -> the model's final answer is
submitted via env.step({"answer": ...}) and rewarded by the env's verifier (calibrated LLM judge by
default for judgment items; the study's numeric matcher for numeric items).

Transcripts + the full per-step trajectory go to gymnasium_transcripts/ (separate from the study's
transcripts/raw/); per-episode rewards to results/gym_scored.csv; heatmap to
results/figures/gym_heatmap_1_iteration.png (+ gym_heatmap_<N>_iterations.png with repeats).

    fh-gym                                   # all models x conditions x items, 1 episode each
    fh-gym --repeats 3 --workers 8
    fh-gym --models opus --conditions tool --items AIR- --limit 5
    fh-gym --reward keyword                  # old keyword verifier instead of the judge
    fh-gym --report                          # rebuild csv + heatmap from stored episodes, no API calls

The no-tool condition is a one-step episode (excerpts in the prompt, answer immediately), kept so
env rewards line up with the study's two conditions.
"""
import argparse
import csv
import json
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

from .. import models
from ..analysis import iteration_heatmaps
from ..paths import GYM_TRANSCRIPTS, RESULTS
from ..prompts import SYSTEM_PROMPT
from ..runner import MODELS, PROVIDER, TOOL_CONDITIONS, load_items, select, user_prompt, write_atomic
from ..tools import TOOL_SPECS
from .environment import FinanceHeadroomEnv

GYM_CSV = RESULTS / "gym_scored.csv"


def episode_path(model_key, tool_condition, item_id, repeat=1):
    suffix = "" if repeat == 1 else f"_r{repeat}"
    return GYM_TRANSCRIPTS / f"{model_key}_{tool_condition}_{item_id}{suffix}.json"


def run_episode(model_key: str, tool_condition: str, item: dict, repeat: int, judge=None, caller=None):
    """One live episode. The env owns tool execution and reward; the model only chooses actions."""
    env = FinanceHeadroomEnv(judge=judge)
    obs, _ = env.reset(options={"item_id": item["id"]})
    steps = []

    def env_executor(tool_name, tool_input):
        # the model's tool call becomes an env action; the env runs the tool and charges step cost
        obs_, reward, terminated, truncated, info = env.step({"tool": tool_name, "args": tool_input})
        steps.append({"action": {"tool": tool_name, "args": tool_input}, "reward": reward,
                      "truncated": truncated, **({"info": info} if info else {})})
        if truncated:
            return "Step limit reached: submit your final answer now."
        return obs_["tool_results"][-1]

    caller = caller or models.CALLERS[model_key]
    prompt = user_prompt(tool_condition, item)
    if tool_condition == "tool":
        final_text, transcript = caller(SYSTEM_PROMPT, prompt, tools=TOOL_SPECS, tool_executor=env_executor)
    else:
        final_text, transcript = caller(SYSTEM_PROMPT, prompt, tools=None, tool_executor=None)

    _, reward, terminated, truncated, info = env.step({"answer": final_text})
    steps.append({"action": {"answer": final_text[-300:]}, "reward": reward, "info": info})
    episode = {
        "model": model_key, "tool_condition": tool_condition, "item_id": item["id"], "repeat": repeat,
        "bucket": env._item.get("bucket", ""), "final_answer": final_text, "transcript": transcript,
        "episode": {
            "verifier": "judge" if judge else "keyword/numeric",
            "judge_model": getattr(judge, "model", None),
            "steps": steps,
            "tool_steps": len(steps) - 1,
            "terminal_reward": reward,
            "return": round(sum(s["reward"] for s in steps), 4),
            "solved": reward == 1.0,
            "scoring": info,
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    }
    write_atomic(episode_path(model_key, tool_condition, item["id"], repeat), json.dumps(episode, indent=2, default=str))
    return episode


def report():
    """Per-episode csv + summary table + heatmap, rebuilt from gymnasium_transcripts/ (no API calls)."""
    eps = [json.loads(p.read_text()) for p in sorted(GYM_TRANSCRIPTS.glob("*.json"))]
    if not eps:
        print("no episodes yet")
        return
    rows = [{"model": e["model"], "tool_condition": e["tool_condition"], "item_id": e["item_id"],
             "repeat": e["repeat"], "bucket": e["bucket"], "verifier": e["episode"]["verifier"],
             "tool_steps": e["episode"]["tool_steps"], "terminal_reward": e["episode"]["terminal_reward"],
             "return": e["episode"]["return"], "solved": e["episode"]["solved"],
             "scoring": e["episode"]["scoring"].get("scoring", "")} for e in eps]
    RESULTS.mkdir(parents=True, exist_ok=True)
    with GYM_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by_mc = defaultdict(list)
    for r in rows:
        by_mc[(r["model"], r["tool_condition"])].append(r)
    print(f"{len(rows)} episodes -> {GYM_CSV}")
    print(f"{'model':<8} {'tool':<8} {'episodes':>8} {'solved':>8} {'mean return':>12} {'mean tool steps':>16}")
    for (m, c), rs in sorted(by_mc.items()):
        solved = sum(r["solved"] for r in rs)
        print(f"{m:<8} {c:<8} {len(rs):>8} {solved / len(rs):>8.1%} {sum(r['return'] for r in rs) / len(rs):>12.3f}"
              f" {sum(r['tool_steps'] for r in rs) / len(rs):>16.1f}")
    try:
        verifiers = ", ".join(sorted({r["verifier"] for r in rows}))
        for out in iteration_heatmaps(rows, "gym_heatmap", f"Live Gymnasium episodes: share solved (verifier: {verifiers})",
                                      ok=lambda r: r["solved"]):
            print(f"wrote {out}")
    except ImportError:
        print("matplotlib not installed -- skipped figure")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fh-gym", description=__doc__.split("\n")[0])
    ap.add_argument("--models", help=f"comma-separated subset of {','.join(MODELS)}")
    ap.add_argument("--conditions", help=f"comma-separated subset of {','.join(TOOL_CONDITIONS)}")
    ap.add_argument("--items", help="comma-separated item-id prefixes")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4, help="concurrent episodes per provider")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--reward", choices=["judge", "keyword"], default="judge",
                    help="verifier for judgment items (numeric items always use the study's matcher)")
    ap.add_argument("--report", action="store_true", help="only rebuild csv + heatmap from stored episodes")
    args = ap.parse_args(argv)
    if args.report:
        return report()

    judge = None
    if args.reward == "judge":
        from ..judge import Judge, calibration_ok
        ok, msg = calibration_ok(Judge().model)
        if not ok:
            raise SystemExit(f"refusing to reward with an uncalibrated judge: {msg} (or use --reward keyword)")
        print(msg)
        judge = Judge()

    GYM_TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
    jobs = select(load_items(), args.models, args.conditions, args.items, args.repeats)
    pending = [j for j in jobs if not episode_path(j[0], j[1], j[2]["id"], j[3]).exists()]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"{len(jobs)} episodes requested: {len(jobs) - len(pending)} already done, {len(pending)} to run")

    gates = {p: threading.Semaphore(args.workers) for p in set(PROVIDER.values())}
    lock, failures, done, started = threading.Lock(), [], 0, time.monotonic()

    def job(m, c, item, r):
        with gates[PROVIDER[m]]:
            return run_episode(m, c, item, r, judge=judge)

    with ThreadPoolExecutor(max_workers=args.workers * len(gates)) as pool:
        futures = {pool.submit(job, *j): j for j in pending}
        for fut in as_completed(futures):
            m, c, item, r = futures[fut]
            with lock:
                try:
                    ep = fut.result()["episode"]
                    done += 1
                    status = f"return {ep['return']:+.2f} ({ep['tool_steps']} tool steps)"
                except Exception as e:  # one failed episode must not stop the rest
                    failures.append({"model": m, "tool_condition": c, "item_id": item["id"], "repeat": r,
                                     "error": f"{type(e).__name__}: {e}"})
                    status = f"FAILED ({type(e).__name__})"
                print(f"[{done + len(failures)}/{len(pending)}] {m} {c} {item['id']} r{r}: {status}"
                      f"  ({time.monotonic() - started:.0f}s)", flush=True)

    write_atomic(RESULTS / "gym_failures.jsonl", "".join(json.dumps(f) + "\n" for f in failures))
    print(f"finished: {done} episodes, {len(failures)} failed" + (" -- see results/gym_failures.jsonl" if failures else ""))
    report()


if __name__ == "__main__":
    main()
