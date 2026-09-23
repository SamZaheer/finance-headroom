"""Replays real transcripts from transcripts/raw through the env, to demonstrate
the env end-to-end using real data and confirm its rewards agree with the study's
own hand-verified results in results/scored.csv (not just synthetic examples).

Run: fh-demo
"""
import csv
import json

from ..paths import SCORED, TRANSCRIPTS
from .environment import FinanceHeadroomEnv


def _scored_lookup() -> dict:
    return {
        (r["model"], r["tool_condition"], r["item_id"]): r
        for r in csv.DictReader(SCORED.open())
    }


def replay(transcript_name: str, env: FinanceHeadroomEnv, scored: dict):
    run = json.loads((TRANSCRIPTS / transcript_name).read_text())
    env.reset(options={"item_id": run["item_id"]})
    for turn in run["transcript"]:
        if isinstance(turn, dict) and (turn.get("tool") or turn.get("name")):
            tool = turn.get("tool") or turn.get("name")
            args = turn.get("input", {})
            env.step({"tool": tool, "args": args})
    _, reward, _, _, info = env.step({"answer": run["final_answer"]})

    key = (run["model"], run["tool_condition"], run["item_id"])
    study_row = scored.get(key, {})
    print(f"{transcript_name:35s} env_reward={reward:>4.1f}  info={info}")
    print(f"{'':35s} study said: auto_correct={study_row.get('auto_correct')!r} "
          f"comparability_flag={study_row.get('comparability_flag')!r}")
    print()


def main():
    env = FinanceHeadroomEnv()
    scored = _scored_lookup()
    # a mix: a numeric control item, a hardened comparability-break item (tool use),
    # and a judgment item the study graded as "partial" (AIR-XENT-01) -- shows all three reward paths
    for name in ["claude_no_tool_TECH-CTRL-04.json", "gpt_tool_MULTI-BREAK-01.json", "gpt_tool_AIR-XENT-01.json"]:
        replay(name, env, scored)


if __name__ == "__main__":
    main()
