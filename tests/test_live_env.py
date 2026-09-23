"""Live env episodes with a fake model: tool calls must be executed by the env (step cost applied),
the final answer rewarded by the env, and the episode stored under gymnasium_transcripts/."""
import json

from finance_headroom.env import live


def fake_caller(answer, tool_calls=()):
    def call(system, prompt, tools=None, tool_executor=None):
        transcript = [{"role": "user", "content": prompt}]
        for name, args in tool_calls:
            out = tool_executor(name, args)  # the env runs the tool
            transcript.append({"role": "tool_result", "tool": name, "input": args, "output": out})
        return answer, transcript
    return call


def test_tool_episode_routes_tools_through_env_and_rewards_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "GYM_TRANSCRIPTS", tmp_path)
    item = {"id": "TECH-CTRL-01", "question": "What was Apple's gross margin in FY2024?"}
    calls = [("search_corpus", {"query": "gross margin net sales"}), ("python_eval", {"code": "print(180683/391035)"})]
    ep = live.run_episode("claude", "tool", item, 1, caller=fake_caller("ANSWER: 46.2%", calls))
    e = ep["episode"]
    assert e["tool_steps"] == 2 and e["terminal_reward"] == 1.0 and e["solved"]
    assert e["return"] == 1.0 - 0.02  # two step costs
    assert "0.462" in ep["transcript"][2]["output"]  # env actually executed the calculator
    stored = json.loads((tmp_path / "claude_tool_TECH-CTRL-01.json").read_text())
    assert stored["episode"]["steps"][0]["action"]["tool"] == "search_corpus"


def test_wrong_answer_and_no_tool_episode(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "GYM_TRANSCRIPTS", tmp_path)
    item = {"id": "TECH-CTRL-01", "question": "q"}
    ep = live.run_episode("gpt", "no_tool", item, 2, caller=fake_caller("ANSWER: $180,683 million"))
    assert ep["episode"]["tool_steps"] == 0 and ep["episode"]["terminal_reward"] == 0.0
    assert (tmp_path / "gpt_no_tool_TECH-CTRL-01_r2.json").exists()


def test_stage2_builds_and_uses_its_own_gold_without_touching_stage1(tmp_path, monkeypatch):
    import csv

    from finance_headroom import judge

    gym = tmp_path / "results" / "gym"
    monkeypatch.setattr(live, "GYM_TRANSCRIPTS", tmp_path / "episodes")
    monkeypatch.setattr(live, "GYM_RESULTS", gym)
    monkeypatch.setattr(live, "GYM_DRAFT", gym / "grading_draft.csv")
    monkeypatch.setattr(live, "GYM_GOLD", tmp_path / "gym_gold.jsonl")
    stage1_cal = tmp_path / "results" / "judge_calibration.json"
    monkeypatch.setattr(judge, "CALIBRATION_JSON", stage1_cal)

    class FakeJudge:
        model = "fake/judge"

        def grade(self, key, answer, tools_used, calls):
            flag = "correct" if "Delta" in answer else "partial"
            return {"comparability_flag": flag, "confidence": "high", "evidence_quote": "q",
                    "evidence_grounding": "cites_relevant_note", "tool_use_correctness": "n/a", "failure_category": ""}

    # 1. bootstrap episodes: judge-rewarded, labelled uncalibrated
    item = {"id": "AIR-CTRL-01", "question": "q"}
    for r, ans in ((1, "ANSWER: Delta"), (2, "ANSWER: United")):
        ep = live.run_episode("gpt", "no_tool", item, r, judge=FakeJudge(), caller=fake_caller(ans),
                              verifier="judge (uncalibrated)")
        assert ep["episode"]["verifier"] == "judge (uncalibrated)"

    # 2. draft -> human approves (and corrects one flag) -> freeze
    live.draft_gold()
    rows = list(csv.DictReader(live.GYM_DRAFT.open()))
    assert {r["judge_flag"] for r in rows} == {"correct", "partial"}
    for r in rows:
        r["approved"] = "yes"
    with live.GYM_DRAFT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=live.DRAFT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    live.freeze_gold()
    gold = judge.load_gold(live.GYM_GOLD)
    assert len(gold) == 2 and all(g["final_answer"] for g in gold)

    # 3. Stage 2 calibrates in its own folder, against its own gold; Stage 1 files never appear
    ok, _ = judge.ensure_calibrated(FakeJudge(), gold=live.GYM_GOLD, workdir=gym)
    assert ok and (gym / "judge_calibration.json").exists()
    assert not stage1_cal.exists()
