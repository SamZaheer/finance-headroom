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
