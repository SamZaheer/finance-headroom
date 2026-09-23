"""Gymnasium environment wrapping the finance-headroom dataset as an RL-shaped task.

This is a REFERENCE PROTOTYPE, not a production environment -- it exists to make the
"Training/RL opportunity" section of report/brief.md concrete: what would it actually
look like to train against this dataset instead of just evaluating it. It reuses the
existing tool implementations and outcome-scoring logic from finance_headroom.tools / .scoring verbatim (imported,
never copied or edited) so the environment's behavior stays consistent with the study's
own results.

Action (a dict, agent-supplied):
    {"tool": "search_corpus" | "python_eval", "args": {...}}   -- call a tool
    {"answer": "<final answer text>"}                          -- submit, ends the episode

Observation (a dict):
    {"item_id": str, "question": str, "tool_results": list[str], "tools_available": list[str]}

Reward:
    - `step_cost` (small negative) on every tool call, to discourage stalling.
    - On answer, for numeric items: 1.0 / 0.0 from the SAME tolerance-band matcher used to
      score the actual study (scoring.py's numeric_match) -- an outcome reward.
    - On answer, for judgment items (no single numeric target): the calibrated rubric judge
      when one is passed in (`judge=`, as fh-gym does by default), otherwise a keyword check
      for whether the answer flags the issue the item tests. The keyword check is a baseline
      only; it is gameable by keyword stuffing and is not used for any reported result.

Note on Gymnasium spaces: action/observation are structured Python dicts (tool-use-agent
shape), not fixed-size arrays, so action_space/observation_space below are declared loosely
(gym.spaces.Text placeholders) rather than strictly validated -- this is a known,
accepted simplification for LLM/tool-use Gym environments, not an oversight.
"""
import json

import gymnasium as gym
from gymnasium import spaces

from ..paths import ANSWER_KEYS
from ..scoring import extract_answer_value, numeric_match  # reused, not modified
from ..tools import python_eval, search_corpus  # reused, not modified

# Heuristic proxy for "did the answer flag a comparability/period/entity issue" --
# see the module docstring's caveat on this being a toy stand-in for real grading.
FLAG_KEYWORDS = [
    "not comparable", "not directly comparable", "restated", "discontinued operations",
    "spin-off", "spinoff", "separation", "different calendar", "fiscal year",
    "not the same", "different periods", "not measuring the same", "beat", "meet", "miss",
]
# "fiscal year" is known to cause 2 false positives on the study's transcripts since it's a
# generic phrase, not a comparability signal specifically. Removing it was tried and made
# overall agreement with the study WORSE (8 mismatches vs. 3) -- it was also legitimately
# catching several TRUE flags elsewhere that no other keyword covered. Kept as-is: a
# concrete, empirical demonstration that keyword-list heuristics trade one error type for
# another rather than actually getting more precise, which is the whole point of flagging
# this as a toy, not a real verifier.


def _load_answer_keys() -> dict:
    keys = {}
    for line in ANSWER_KEYS.read_text().splitlines():
        if line.strip():
            ak = json.loads(line)
            keys[ak["id"]] = ak
    return keys


class FinanceHeadroomEnv(gym.Env):
    """One episode = one dataset item (see module docstring for action/observation/reward)."""

    metadata = {"render_modes": []}

    def __init__(self, item_ids=None, max_steps: int = 10, step_cost: float = -0.01, judge=None):
        """judge: optional finance_headroom.judge.Judge. When given, judgment items are rewarded by the
        calibrated LLM judge (1.0 only for "correct") instead of the FLAG_KEYWORDS heuristic."""
        super().__init__()
        self.judge = judge
        self.answer_keys = _load_answer_keys()
        self.item_ids = item_ids or list(self.answer_keys.keys())
        self.max_steps = max_steps
        self.step_cost = step_cost
        self.action_space = spaces.Text(max_length=20000)
        self.observation_space = spaces.Text(max_length=200000)
        self._item = None
        self._steps = 0
        self._tool_log = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        item_id = (options or {}).get("item_id")
        if item_id is None:
            item_id = self.item_ids[self.np_random.integers(len(self.item_ids))]
        self._item = self.answer_keys[item_id]
        self._steps = 0
        self._tool_log = []
        return self._obs(), {"item_id": item_id}

    def step(self, action: dict):
        if "answer" in action:
            reward, info = self._score_answer(action["answer"])
            return self._obs(), reward, True, False, info

        self._steps += 1
        if self._steps > self.max_steps:
            return self._obs(), self.step_cost, False, True, {"reason": "max_steps_exceeded"}

        tool, args = action.get("tool"), action.get("args", {})
        if tool == "search_corpus":
            result = search_corpus(args.get("query", ""), self._item["id"])
        elif tool == "python_eval":
            result = python_eval(args.get("code", ""))
        else:
            result = f"Unknown tool: {tool}"
        self._tool_log.append({"tool": tool, "args": args, "result": result})
        return self._obs(), self.step_cost, False, False, {}

    def _obs(self) -> dict:
        return {
            "item_id": self._item["id"],
            "question": self._item["question"],
            "tool_results": [t["result"] for t in self._tool_log],
            "tools_available": ["search_corpus", "python_eval"],
        }

    def _score_answer(self, answer_text: str):
        expected = self._item.get("expected_final_answer", {})

        # numeric items: score only the declared "ANSWER:" line, exactly like scoring.py
        # does -- scoring the whole raw response would give undeserved credit when the
        # model's reasoning mentions the correct number in passing before it states a
        # wrong final answer (this was a real bug caught by replaying gpt_tool_TECH-CTRL-03).
        declared = extract_answer_value(answer_text)
        numeric_ok = numeric_match(declared, expected)
        if numeric_ok is not None:
            return (1.0 if numeric_ok else 0.0), {"scoring": "numeric", "correct": numeric_ok}

        if self.judge is not None:
            calls = [{"tool": t["tool"], "input": t["args"], "output": str(t["result"])} for t in self._tool_log]
            grade = self.judge.grade(self._item, answer_text, bool(calls), calls)
            ok = grade["comparability_flag"] == "correct"
            return (1.0 if ok else 0.0), {"scoring": "judge", "flag": grade["comparability_flag"],
                                          "confidence": grade["confidence"]}

        # judgment items: check the FULL response, matching how the study's manual grading
        # actually worked (reading the whole transcript, not just the final line)
        flagged = any(kw in answer_text.lower() for kw in FLAG_KEYWORDS)
        return (1.0 if flagged else 0.0), {"scoring": "heuristic_flag_keyword", "flagged": flagged}
