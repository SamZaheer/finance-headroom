"""Self-checks for the env's reward branches."""
from finance_headroom.env import FinanceHeadroomEnv


def test_numeric_item_correct_and_incorrect():
    env = FinanceHeadroomEnv()
    env.reset(options={"item_id": "TECH-CTRL-01"})  # Apple gross margin, expects ~46.2%
    _, reward, terminated, _, info = env.step({"answer": "ANSWER: 46.2%"})
    assert terminated and reward == 1.0 and info["scoring"] == "numeric", info

    env.reset(options={"item_id": "TECH-CTRL-01"})
    _, reward, terminated, _, info = env.step({"answer": "ANSWER: 12.0%"})
    assert terminated and reward == 0.0, info


def test_judgment_item_flag_keyword_heuristic():
    env = FinanceHeadroomEnv()
    env.reset(options={"item_id": "MULTI-BREAK-01"})  # IBM/Kyndryl comparability item, judgment-scored
    _, reward, terminated, _, info = env.step(
        {"answer": "IBM's revenue grew because the restated, continuing-operations figure "
                    "excludes Kyndryl, which is not comparable to the original figure."}
    )
    assert terminated and reward == 1.0 and info["flagged"] is True, info

    env.reset(options={"item_id": "MULTI-BREAK-01"})
    _, reward, terminated, _, info = env.step({"answer": "Revenue grew 10%."})
    assert terminated and reward == 0.0 and info["flagged"] is False, info


def test_tool_call_step_then_answer():
    env = FinanceHeadroomEnv(max_steps=5)
    obs, _ = env.reset(options={"item_id": "TECH-CTRL-01"})
    assert obs["tool_results"] == []
    obs, reward, terminated, truncated, _ = env.step(
        {"tool": "search_corpus", "args": {"query": "gross margin"}}
    )
    assert not terminated and not truncated and reward == env.step_cost
    assert len(obs["tool_results"]) == 1


def test_max_steps_truncates():
    env = FinanceHeadroomEnv(max_steps=2)
    env.reset(options={"item_id": "TECH-CTRL-01"})
    env.step({"tool": "search_corpus", "args": {"query": "x"}})
    env.step({"tool": "search_corpus", "args": {"query": "x"}})
    _, _, terminated, truncated, info = env.step({"tool": "search_corpus", "args": {"query": "x"}})
    assert truncated and not terminated and info["reason"] == "max_steps_exceeded"

