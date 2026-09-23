"""Judge (validation, cache, calibration gate) and Batch API path (submit / wait / collect / resume),
with fake clients -- no API calls."""
import json
from types import SimpleNamespace as NS

import pytest

from finance_headroom import batch, judge, runner
from tests.test_runner import _setup


def test_validate_downgrades_unverifiable_quotes_and_clears_failures_on_correct():
    answer = "Delta leads on margin, but Alaska grew fastest."
    g = judge.validate({"comparability_flag": "correct", "evidence_quote": "Alaska grew fastest", "confidence": "high",
                        "failure_category": ["computation"], "tool_use_correctness": "effective", "rationale": "ok"},
                       answer, tools_used=False)
    assert g["confidence"] == "high" and g["failure_category"] == "" and g["tool_use_correctness"] == "n/a"
    g = judge.validate({"comparability_flag": "partial", "evidence_quote": "invented words", "confidence": "high",
                        "failure_category": ["unsupported_conclusion"]}, answer, tools_used=True)
    assert g["confidence"] == "low" and "quote not found" in g["evidence_quote"] and g["failure_category"] == "unsupported_conclusion"
    with pytest.raises(ValueError):
        judge.validate({"comparability_flag": "maybe"}, answer, tools_used=False)


def test_judge_caches_by_answer_and_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(judge, "CACHE", tmp_path / "cache.jsonl")
    monkeypatch.setattr(judge, "RESULTS", tmp_path)
    calls = []

    def create(**kw):
        calls.append(kw)
        body = {"comparability_flag": "correct", "evidence_grounding": "cites_relevant_note", "tool_use_correctness": "n/a",
                "failure_category": [], "evidence_quote": "Delta", "confidence": "high", "rationale": "fine"}
        return NS(choices=[NS(message=NS(content=json.dumps(body)))])

    client = NS(chat=NS(completions=NS(create=create)))
    key = {"id": "X-01", "question": "q", "grading_rubric": "r"}
    j = judge.Judge("fake/judge", client=client)
    assert j.grade(key, "Delta wins", False, [])["comparability_flag"] == "correct"
    j.grade(key, "Delta wins", False, [])
    assert len(calls) == 1
    assert calls[0]["response_format"]["type"] == "json_schema" and calls[0]["temperature"] == 0
    judge.Judge("fake/judge", client=client).grade(key, "Delta wins", False, [])  # fresh instance, same disk cache
    assert len(calls) == 1
    judge.Judge("fake/judge", client=client).grade(key, "United wins", False, [])  # different answer -> new call
    assert len(calls) == 2


def test_apply_refuses_without_a_passing_calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(judge, "CALIBRATION_JSON", tmp_path / "cal.json")
    assert judge.calibration_ok("m")[0] is False
    (tmp_path / "cal.json").write_text(json.dumps({"judge_model": "m", "prompt_hash": judge.prompt_hash(),
                                                   "passed": False, "agreement": 0.8, "threshold": 0.9}))
    assert judge.calibration_ok("m")[0] is False
    (tmp_path / "cal.json").write_text(json.dumps({"judge_model": "m", "prompt_hash": judge.prompt_hash(),
                                                   "passed": True, "agreement": 0.95, "threshold": 0.9}))
    assert judge.calibration_ok("m")[0] is True
    assert judge.calibration_ok("other-model")[0] is False  # calibration doesn't transfer across judges


class FakeBatches:
    """Enough of client.messages.batches: create, retrieve (ends after `polls` checks), results."""

    def __init__(self, polls=1):
        self.created, self.polls, self.checks = [], polls, 0

    def create(self, requests):
        self.created.append(list(requests))
        return NS(id=f"batch_{len(self.created)}")

    def retrieve(self, batch_id):
        self.checks += 1
        status = "ended" if self.checks >= self.polls else "in_progress"
        return NS(processing_status=status, request_counts=NS(succeeded=0, processing=1, errored=0))

    def results(self, batch_id):
        reqs = self.created[int(batch_id.split("_")[1]) - 1]
        for r in reqs:
            if r["custom_id"] == "claude__no_tool__T-01__r1":
                yield NS(custom_id=r["custom_id"], result=NS(type="errored", error="overloaded"))
                continue
            block = NS(type="text", text="ANSWER: 1", model_dump=lambda: {"type": "text", "text": "ANSWER: 1"})
            yield NS(custom_id=r["custom_id"], result=NS(type="succeeded", message=NS(content=[block])))


def _batch_setup(tmp_path, monkeypatch, polls=1):
    _setup(tmp_path, monkeypatch, n_items=2)
    fake = FakeBatches(polls)
    monkeypatch.setattr(batch, "BATCH_DIR", tmp_path / "results" / "batches")
    monkeypatch.setattr(batch, "default_client", lambda: NS(messages=NS(batches=fake)))
    return fake


def test_batch_routes_only_anthropic_no_tool_jobs_and_collects(tmp_path, monkeypatch):
    fake = _batch_setup(tmp_path, monkeypatch)
    runner.main(["--batch", "--poll-seconds", "0"])
    batched = {r["custom_id"] for r in fake.created[0]}
    assert batched == {f"{m}__no_tool__T-0{i}__r1" for m in ("claude", "opus") for i in (0, 1)}
    req = fake.created[0][0]["params"]
    assert req["tools"] == [] and req["max_tokens"] == 4096 and "<excerpts>" in req["messages"][0]["content"]
    raw = tmp_path / "raw"
    assert (raw / "claude_no_tool_T-00.json").exists() and (raw / "gpt_no_tool_T-00.json").exists()
    assert not (raw / "claude_no_tool_T-01.json").exists()  # errored in batch -> logged, not written
    failures = (tmp_path / "results" / "run_failures.jsonl").read_text()
    assert "claude" in failures and "batch errored" in failures
    manifest = json.loads((tmp_path / "results" / "run_manifest.json").read_text())
    assert manifest["batched"] == 4 and manifest["succeeded"] == 3 + 8  # 3 batch + 8 real-time (gpt x2, tool x6)


def test_no_wait_then_collect_never_resubmits(tmp_path, monkeypatch):
    fake = _batch_setup(tmp_path, monkeypatch, polls=2)
    runner.main(["--batch", "--no-wait", "--models", "opus", "--conditions", "no_tool"])
    assert len(fake.created) == 1 and not list((tmp_path / "raw").glob("*.json"))
    runner.main(["--batch", "--no-wait", "--models", "opus", "--conditions", "no_tool"])  # still in flight
    assert len(fake.created) == 1
    runner.main(["--collect"])  # first check: still running
    assert not list((tmp_path / "raw").glob("*.json"))
    runner.main(["--collect"])  # ended -> written
    assert sorted(p.name for p in (tmp_path / "raw").glob("*.json")) == ["opus_no_tool_T-00.json", "opus_no_tool_T-01.json"]
    rec = json.loads(next((tmp_path / "results" / "batches").glob("*.json")).read_text())
    assert rec["status"] == "collected"


def test_rerunning_apply_never_auto_applies_rows_queued_for_review(tmp_path, monkeypatch):
    import csv
    for name, fname in (("SCORED", "scored.csv"), ("JUDGE_CSV", "judge.csv"), ("DRAFT_CSV", "draft.csv"),
                        ("CALIBRATION_JSON", "cal.json")):
        monkeypatch.setattr(judge, name, tmp_path / fname)
    (tmp_path / "cal.json").write_text(json.dumps({"judge_model": "m", "prompt_hash": judge.prompt_hash(), "passed": True,
                                                   "agreement": 0.99, "threshold": 0.9}))
    base = ["model", "tool_condition", "item_id", "repeat", "bucket", "auto_correct", *judge.GRADE_COLUMNS]
    rows = [{"model": "gpt", "tool_condition": "tool", "item_id": f"X-{i:02d}", "repeat": "1", "auto_correct": ""}
            for i in range(60)]
    judge.write_csv(tmp_path / "scored.csv", rows, base)
    grades = [{**r, "comparability_flag": "correct", "evidence_grounding": "cites_relevant_note", "tool_use_correctness": "n/a",
               "failure_category": "", "evidence_quote": "q", "confidence": "high"} for r in rows]
    judge.write_csv(tmp_path / "judge.csv", grades, [*base, "confidence"])

    judge.main(["--apply", "--model", "m", "--seed", "1"])
    queued = {d["item_id"] for d in csv.DictReader((tmp_path / "draft.csv").open())}
    assert queued  # some rows were spot-checked
    for seed in ("2", "3"):  # re-running with a different sample must not release them
        judge.main(["--apply", "--model", "m", "--seed", seed])
    applied = {r["item_id"] for r in csv.DictReader((tmp_path / "scored.csv").open()) if r["comparability_flag"]}
    assert not (queued & applied)
    assert len(applied) + len(queued) == 60


def test_judge_retries_malformed_json(tmp_path, monkeypatch):
    monkeypatch.setattr(judge, "CACHE", tmp_path / "cache.jsonl")
    monkeypatch.setattr(judge, "RESULTS", tmp_path)
    good = {"comparability_flag": "partial", "evidence_grounding": "uncited", "tool_use_correctness": "n/a",
            "failure_category": [], "evidence_quote": "Delta", "confidence": "high", "rationale": "r"}
    replies = iter(['{"comparability_flag": "par', json.dumps(good)])
    client = NS(chat=NS(completions=NS(create=lambda **kw: NS(choices=[NS(message=NS(content=next(replies)))]))))
    assert judge.Judge("m", client=client).grade({"id": "X"}, "Delta", False, [])["comparability_flag"] == "partial"
