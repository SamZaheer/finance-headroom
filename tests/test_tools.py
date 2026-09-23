"""Self-checks for the two parsers that matter: corpus search ranking and
answer-value tolerance matching."""
from finance_headroom import scoring, tools


def test_search_corpus_ranks_relevant_paragraph_first(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "CORPUS_DIR", tmp_path)
    item_dir = tmp_path / "X-01"
    item_dir.mkdir()
    (item_dir / "doc1.txt").write_text(
        "Irrelevant paragraph about the weather.\n\n"
        "The Kyndryl separation was completed in November 2021 and is "
        "presented as discontinued operations.\n\n"
        "Another irrelevant paragraph."
    )
    result = tools.search_corpus("Kyndryl discontinued operations", "X-01")
    assert "Kyndryl" in result, result
    assert "Irrelevant" not in result, "zero-score paragraphs should be filtered out"


def test_numeric_match_within_and_outside_tolerance():
    expected = {"value_range": [10.0, 10.0], "tolerance": 0.5}
    assert scoring.numeric_match("ANSWER: 10.3%", expected) is True
    assert scoring.numeric_match("ANSWER: 12.0%", expected) is False


def test_numeric_match_finds_value_after_intermediate_numbers():
    # regression: models often state an intermediate value before the final answer
    # ("from 7.80% to 8.02%, an increase of 0.22pp") -- must not grab only the first number
    expected = {"value_range": [0.21, 0.21], "tolerance": 0.1}
    answer = "R&D intensity rose from 7.80% in FY2023 to 8.02% in FY2024, an increase of 0.22 percentage points."
    assert scoring.numeric_match(answer, expected) is True


def test_numeric_match_honors_stated_scale_for_dollar_millions():
    expected = {"value_range": [768.044, 768.044], "unit": "USD millions", "tolerance": 1}
    assert scoring.numeric_match("$768,044 thousand", expected) is True
    assert scoring.numeric_match("$768,044", expected) is False  # unlabeled thousands stays wrong
    big = {"value_range": [637959, 637959], "unit": "USD millions", "tolerance": 100}
    assert scoring.numeric_match("$637.959 billion in net sales", big) is True
    assert scoring.numeric_match("$637,959 million", big) is True
    # scale words never rescue a percent answer given as dollars (margin-vs-profit slip)
    pct = {"value_range": [46.2, 46.2], "unit": "percent", "tolerance": 0.3}
    assert scoring.numeric_match("$180,683 million", pct) is False


def test_numeric_match_returns_none_for_judgment_items():
    assert scoring.numeric_match("some text answer", {}) is None
    # regression: real dataset uses {"value_range": None, ...} (not [None, None]) for judgment items
    assert scoring.numeric_match("some text answer", {"value_range": None, "unit": "text"}) is None


def test_display_name_from_model_ids():
    from finance_headroom.analysis import display_name
    assert display_name("claude-sonnet-5") == "Claude Sonnet 5"
    assert display_name("claude-opus-5") == "Claude Opus 5"
    assert display_name("openai/gpt-5.1") == "GPT-5.1"
