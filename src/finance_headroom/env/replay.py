"""Replays EVERY transcript in transcripts/raw through the env and aggregates the
env's reward by (model, tool_condition, bucket) -- the same grouping the main study uses
in results/scored.csv -- to check the env's reward function agrees with the study's own
hand-graded results at full scale, not just the 3-transcript demo.

Writes results/env_scored.csv and results/figures/env_reward_heatmap.png; the main
study's scored.csv and study heatmaps are untouched.

Run: fh-replay            # keyword heuristic for judgment items (free, offline)
     fh-replay --judge    # calibrated LLM judge instead (API calls, cached) -> *_judge files
"""
import argparse
import csv
import json
from collections import defaultdict

from ..analysis import plot_heatmap
from ..paths import FIGURES, RESULTS, SCORED, TRANSCRIPTS
from .environment import FinanceHeadroomEnv


def replay_one(env: FinanceHeadroomEnv, run: dict):
    env.reset(options={"item_id": run["item_id"]})
    for turn in run["transcript"]:
        if isinstance(turn, dict) and (turn.get("tool") or turn.get("name")):
            tool = turn.get("tool") or turn.get("name")
            args = turn.get("input", {})
            env.step({"tool": tool, "args": args})
    _, reward, _, _, info = env.step({"answer": run["final_answer"]})
    return reward, info


def collect(judge=None):
    env = FinanceHeadroomEnv(judge=judge)
    rows = []
    for path in sorted(TRANSCRIPTS.glob("*.json")):
        run = json.loads(path.read_text())
        bucket = env.answer_keys[run["item_id"]]["bucket"]
        reward, info = replay_one(env, run)
        rows.append({
            "model": run["model"], "tool_condition": run["tool_condition"],
            "item_id": run["item_id"], "repeat": run.get("repeat", 1), "bucket": bucket,
            "env_reward": reward, "scoring": info.get("scoring", ""),
        })
    return rows


def write_csv(rows, out_path=RESULTS / "env_scored.csv"):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")


def summarize(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["model"], r["tool_condition"], r["bucket"])].append(r["env_reward"])
    summary = {k: sum(v) / len(v) for k, v in grouped.items()}
    print(f"{'model':<8} {'tool':<10} {'bucket':<25} {'n':>3} {'env_accuracy':>12}")
    for k in sorted(summary):
        print(f"{k[0]:<8} {k[1]:<10} {k[2]:<25} {len(grouped[k]):>3} {summary[k]:>11.0%}")
    return summary


def agreement(rows):
    """Env reward vs. the study's own grades in scored.csv (numeric auto-score or human grade)."""
    from ..analysis import is_correct
    study = {(r["model"], r["tool_condition"], r["item_id"], r.get("repeat") or "1"): r for r in csv.DictReader(SCORED.open())}
    pairs = [(r, study.get((r["model"], r["tool_condition"], r["item_id"], str(r["repeat"])))) for r in rows]
    pairs = [(r, s) for r, s in pairs if s and (s["auto_correct"] or s["comparability_flag"])]
    for label, sel in (("all", pairs), ("judgment", [p for p in pairs if not p[1]["auto_correct"]])):
        agree = sum((r["env_reward"] == 1.0) == is_correct(s) for r, s in sel)
        print(f"env vs study grades ({label}): {agree}/{len(sel)} = {agree / len(sel):.1%}" if sel else "")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fh-replay")
    ap.add_argument("--judge", action="store_true", help="reward judgment items with the calibrated LLM judge")
    args = ap.parse_args(argv)
    judge, tag = None, ""
    if args.judge:
        from ..judge import Judge, calibration_ok
        ok, msg = calibration_ok(Judge().model)
        print(("" if ok else "WARNING: ") + msg)
        judge, tag = Judge(), "_judge"
    rows = collect(judge)
    write_csv(rows, RESULTS / f"env_scored{tag}.csv")
    summary = summarize(rows)
    agreement(rows)
    try:
        out = FIGURES / f"env_reward_heatmap{tag}.png"
        verifier = "LLM judge" if judge else "keyword heuristic"
        title = f"gym_env reward by bucket, model, and tool condition\n(replayed from real transcripts; judgment items: {verifier})"
        plot_heatmap(summary, title, out, title_fontsize=12)
        print(f"wrote {out}")
    except ImportError:
        print("matplotlib not installed -- skipped figure")


if __name__ == "__main__":
    main()
