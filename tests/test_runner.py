"""Parallel runner end to end with a fake model (no API calls): per-provider concurrency cap,
resume, repeats, failure logging, and repeat-aware grade preservation in scoring."""
import csv
import json
import threading
import time

from finance_headroom import runner, scoring


def _setup(tmp_path, monkeypatch, n_items=6):
    q = tmp_path / "questions.jsonl"
    q.write_text("".join(json.dumps({"id": f"T-{i:02d}", "question": "q"}) + "\n" for i in range(n_items)))
    (tmp_path / "corpus").mkdir()
    for i in range(n_items):
        (tmp_path / "corpus" / f"T-{i:02d}").mkdir()
    monkeypatch.setattr(runner, "QUESTIONS", q)
    monkeypatch.setattr(runner, "CORPUS", tmp_path / "corpus")
    monkeypatch.setattr(runner, "TRANSCRIPTS", tmp_path / "raw")
    monkeypatch.setattr(runner, "RESULTS", tmp_path / "results")

    live, peak, lock = {"anthropic": 0, "openai": 0}, {"anthropic": 0, "openai": 0}, threading.Lock()

    def fake_caller(provider):
        def call(system, prompt, tools=None, tool_executor=None):
            with lock:
                live[provider] += 1
                peak[provider] = max(peak[provider], live[provider])
            time.sleep(0.02)
            with lock:
                live[provider] -= 1
            return "ANSWER: 1", [{"role": "user", "content": prompt}]
        return call

    monkeypatch.setattr(runner, "CALLERS", {"claude": fake_caller("anthropic"), "opus": fake_caller("anthropic"),
                                            "gpt": fake_caller("openai")})
    return peak


def test_parallel_run_respects_per_provider_cap_and_resumes(tmp_path, monkeypatch):
    peak = _setup(tmp_path, monkeypatch)
    runner.main(["--workers", "2", "--repeats", "2"])
    files = sorted(p.name for p in (tmp_path / "raw").glob("*.json"))
    assert len(files) == 3 * 2 * 6 * 2
    assert "claude_tool_T-00.json" in files and "claude_tool_T-00_r2.json" in files  # r1 keeps the old name
    assert peak["anthropic"] <= 2 and peak["openai"] <= 2
    assert json.loads((tmp_path / "raw" / "gpt_no_tool_T-03_r2.json").read_text())["repeat"] == 2
    manifest = json.loads((tmp_path / "results" / "run_manifest.json").read_text())
    assert manifest["succeeded"] == 72 and manifest["failed"] == 0

    runner.main(["--workers", "2", "--repeats", "2"])  # everything exists -> nothing to run
    assert json.loads((tmp_path / "results" / "run_manifest.json").read_text())["ran"] == 0


def test_failures_are_logged_and_retried_on_next_run(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, n_items=2)
    calls = runner.CALLERS
    monkeypatch.setattr(runner, "CALLERS", {**calls, "gpt": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("429"))})
    runner.main(["--models", "gpt", "--conditions", "tool"])
    failures = [json.loads(line) for line in (tmp_path / "results" / "run_failures.jsonl").read_text().splitlines()]
    assert len(failures) == 2 and "429" in failures[0]["error"]
    assert not list((tmp_path / "raw").glob("*"))  # nothing half-written

    monkeypatch.setattr(runner, "CALLERS", calls)
    runner.main(["--models", "gpt", "--conditions", "tool"])
    assert len(list((tmp_path / "raw").glob("*.json"))) == 2
    assert (tmp_path / "results" / "run_failures.jsonl").read_text() == ""


def test_select_filters_and_limit(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    jobs = runner.select(runner.load_items(), "opus", "no_tool", "T-01,T-02", 3)
    assert {(m, c, it["id"]) for m, c, it, _ in jobs} == {("opus", "no_tool", "T-01"), ("opus", "no_tool", "T-02")}
    assert len(jobs) == 6
    runner.main(["--limit", "5"])
    assert len(list((tmp_path / "raw").glob("*.json"))) == 5


def test_scoring_keeps_old_grades_as_repeat_1(tmp_path, monkeypatch):
    out = tmp_path / "scored.csv"
    with out.open("w", newline="") as f:  # a pre-repeats scored.csv: no repeat column
        w = csv.DictWriter(f, fieldnames=["model", "tool_condition", "item_id", *scoring.MANUAL_COLUMNS])
        w.writeheader()
        w.writerow({"model": "gpt", "tool_condition": "tool", "item_id": "X-01", "comparability_flag": "partial"})
    monkeypatch.setattr(scoring, "OUT", out)
    grades = scoring.load_existing_manual_grades()
    assert grades[("gpt", "tool", "X-01", "1")]["comparability_flag"] == "partial"
