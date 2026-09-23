"""Repo-relative data/output locations, resolved once. Set FH_ROOT to point elsewhere
(the Docker image sets it to /app)."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("FH_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
CORPUS = DATA / "corpus"
QUESTIONS = DATA / "questions.jsonl"
ANSWER_KEYS = DATA / "answer_keys.jsonl"
TRANSCRIPTS = ROOT / "transcripts" / "raw"
GYM_TRANSCRIPTS = ROOT / "gymnasium_transcripts"  # live env episodes (fh-gym), never mixed with the study's
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
SCORED = RESULTS / "scored.csv"
