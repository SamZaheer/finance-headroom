"""Live Gymnasium episodes (fh-gym): the env drives real LLM calls -- no transcript replay.

Each episode: env.reset(item) -> the model gets the question -> every tool call the model makes is
executed BY THE ENV via env.step({"tool": ...}) (step cost applies) -> the model's final answer is
submitted via env.step({"answer": ...}) and rewarded by the env's verifier (calibrated LLM judge by
default for judgment items; the study's numeric matcher for numeric items).

Stage 2 is self-contained: it never reads Stage 1's transcripts, grades, judge cache or gold set.
Episodes (transcript + per-step trajectory) go to gymnasium_transcripts/; everything else to results/gym/
(gym_scored.csv, judge cache + calibration, grading_draft.csv, figures/gym_heatmap_*.png). The judge is
calibrated against Stage 2's own gold set, data/calibration/gym_gold.jsonl, built from Stage 2 episodes:

    fh-gym                  # no gym gold yet -> runs, rewards labelled "judge (uncalibrated)"
    fh-gym --draft-gold     # judgment episodes + judge verdicts -> results/gym/grading_draft.csv
    #   ...edit human_flag where you disagree, set approved=yes...
    fh-gym --freeze-gold    # approved rows -> data/calibration/gym_gold.jsonl
    fh-gym                  # from now on: auto-calibrates against gym gold, rewards "judge"

    fh-gym                                   # all models x conditions x items, 1 episode each
    fh-gym --repeats 3 --workers 8
    fh-gym --models opus --conditions tool --items AIR- --limit 5
    fh-gym --reward keyword                  # old keyword verifier instead of the judge
    fh-gym --report                          # rebuild csv + heatmap from stored episodes, no API calls
    fh-gym --no-report                       # just run episodes, skip the trailing report() (pair with --report after)

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
from ..paths import GYM_GOLD, GYM_RESULTS, GYM_TRANSCRIPTS
from ..prompts import SYSTEM_PROMPT
from ..runner import MODELS, PROVIDER, TOOL_CONDITIONS, load_items, select, user_prompt, write_atomic
from ..tools import TOOL_SPECS
from .environment import FinanceHeadroomEnv

GYM_CSV = GYM_RESULTS / "gym_scored.csv"
GYM_DRAFT = GYM_RESULTS / "grading_draft.csv"
GYM_CACHE = GYM_RESULTS / "judge_cache.jsonl"


def episode_path(model_key, tool_condition, item_id, repeat=1):
    suffix = "" if repeat == 1 else f"_r{repeat}"
    return GYM_TRANSCRIPTS / f"{model_key}_{tool_condition}_{item_id}{suffix}.json"


def run_episode(model_key: str, tool_condition: str, item: dict, repeat: int, judge=None, caller=None, verifier=None):
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
            "verifier": verifier or ("judge" if judge else "keyword/numeric"),
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
    GYM_RESULTS.mkdir(parents=True, exist_ok=True)
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
                                      ok=lambda r: r["solved"], fig_dir=GYM_RESULTS / "figures"):
            print(f"wrote {out}")
    except ImportError:
        print("matplotlib not installed -- skipped figure")


DRAFT_FIELDS = ["model", "tool_condition", "item_id", "repeat", "bucket", "judge_flag", "judge_confidence",
                "human_flag", "approved", "answer_excerpt"]


def draft_gold():
    """Judgment episodes + the judge's verdict -> results/gym/grading_draft.csv, for a human to confirm or correct
    (edit human_flag, set approved=yes). Rows already in the draft keep their human edits."""
    existing = {(d["model"], d["tool_condition"], d["item_id"], d["repeat"]): d
                for d in csv.DictReader(GYM_DRAFT.open())} if GYM_DRAFT.exists() else {}
    rows = []
    for p in sorted(GYM_TRANSCRIPTS.glob("*.json")):
        e = json.loads(p.read_text())
        info = e["episode"]["scoring"]
        if info.get("scoring") != "judge":
            continue  # numeric items are scored exactly against the answer key; nothing to calibrate
        k = (e["model"], e["tool_condition"], e["item_id"], str(e["repeat"]))
        rows.append(existing.get(k) or {
            "model": e["model"], "tool_condition": e["tool_condition"], "item_id": e["item_id"], "repeat": e["repeat"],
            "bucket": e["bucket"], "judge_flag": info.get("flag", ""), "judge_confidence": info.get("confidence", ""),
            "human_flag": info.get("flag", ""), "approved": "", "answer_excerpt": e["final_answer"][-600:]})
    GYM_RESULTS.mkdir(parents=True, exist_ok=True)
    with GYM_DRAFT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DRAFT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} judgment episodes -> {GYM_DRAFT}: check human_flag (correct/partial/incorrect), set approved=yes")


def freeze_gold():
    """Approved Stage 2 draft rows (+ each episode's full answer and tool log) -> data/calibration/gym_gold.jsonl."""
    from ..judge import FLAGS, load_gold, tool_log
    if not GYM_DRAFT.exists():
        raise SystemExit(f"no {GYM_DRAFT} yet -- run fh-gym --draft-gold first")
    gold = {(g["model"], g["tool_condition"], g["item_id"], str(g["repeat"])): g for g in load_gold(GYM_GOLD)}
    added = 0
    for d in csv.DictReader(GYM_DRAFT.open()):
        if d["approved"].strip().lower() not in ("yes", "y", "true", "1"):
            continue
        if d["human_flag"] not in FLAGS:
            raise SystemExit(f"bad human_flag {d['human_flag']!r} for {d['model']} {d['tool_condition']} {d['item_id']}")
        k = (d["model"], d["tool_condition"], d["item_id"], str(d["repeat"]))
        e = json.loads(episode_path(d["model"], d["tool_condition"], d["item_id"], int(d["repeat"])).read_text())
        tools_used = d["tool_condition"] == "tool"
        added += k not in gold
        gold[k] = {"model": d["model"], "tool_condition": d["tool_condition"], "item_id": d["item_id"],
                   "repeat": str(d["repeat"]), "bucket": d["bucket"], "human_flag": d["human_flag"],
                   "final_answer": e["final_answer"], "tools_used": tools_used,
                   "calls": tool_log(e["transcript"]) if tools_used else []}
    if not gold:
        raise SystemExit(f"no rows marked approved=yes in {GYM_DRAFT}")
    GYM_GOLD.parent.mkdir(parents=True, exist_ok=True)
    GYM_GOLD.write_text("".join(json.dumps(g) + "\n" for g in gold.values()))
    print(f"added {added} approved rows to {GYM_GOLD} ({len(gold)} total); the next fh-gym run calibrates against it")


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
    ap.add_argument("--draft-gold", action="store_true", help="judgment episodes -> results/gym/grading_draft.csv for approval")
    ap.add_argument("--freeze-gold", action="store_true", help="approved draft rows -> data/calibration/gym_gold.jsonl")
    ap.add_argument("--no-report", action="store_true",
                    help="skip the trailing report() after a live run -- run it separately with --report")
    args = ap.parse_args(argv)
    if args.report:
        return report()
    if args.draft_gold:
        return draft_gold()
    if args.freeze_gold:
        return freeze_gold()

    judge, verifier = None, None
    if args.reward == "judge":
        from ..judge import Judge, ensure_calibrated
        judge = Judge(cache=GYM_CACHE)
        ok, msg = ensure_calibrated(judge, args.workers, gold=GYM_GOLD, workdir=GYM_RESULTS)
        if ok:
            verifier = "judge"
            print(msg)
        elif not GYM_GOLD.exists():
            # bootstrap: Stage 2 has no gold set of its own yet. Run, label every reward as uncalibrated, and
            # build the gold set from these episodes afterwards (--draft-gold / --freeze-gold).
            verifier = "judge (uncalibrated)"
            print(f"no Stage 2 gold set yet ({GYM_GOLD}): rewards are labelled 'judge (uncalibrated)'. After this run: "
                  f"fh-gym --draft-gold, approve rows, fh-gym --freeze-gold")
        else:
            raise SystemExit(f"refusing to reward with a judge that failed Stage 2 calibration: {msg} "
                             f"(fix {GYM_GOLD} or use --reward keyword)")

    GYM_TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
    jobs = select(load_items(), args.models, args.conditions, args.items, args.repeats)
    pending = [j for j in jobs if not episode_path(j[0], j[1], j[2]["id"], j[3]).exists()]
    n_done = len(jobs) - len(pending)  # counted before --limit, which only defers work
    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"{len(jobs)} episodes requested: {n_done} already done, {len(pending)} to run now")

    gates = {p: threading.Semaphore(args.workers) for p in set(PROVIDER.values())}
    lock, failures, done, started = threading.Lock(), [], 0, time.monotonic()

    def job(m, c, item, r):
        with gates[PROVIDER[m]]:
            return run_episode(m, c, item, r, judge=judge, verifier=verifier)

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

    write_atomic(GYM_RESULTS / "gym_failures.jsonl", "".join(json.dumps(f) + "\n" for f in failures))
    print(f"finished: {done} episodes, {len(failures)} failed" + (" -- see results/gym_failures.jsonl" if failures else ""))
    if not args.no_report:
        report()


if __name__ == "__main__":
    main()
