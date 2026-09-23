# finance-headroom

Measures whether frontier LLMs (Claude Sonnet 5, Claude Opus 5, GPT-5.1) correctly handle **accounting comparability breaks**, such as spinoffs, discontinued-operations restatements, fiscal-year misalignment and acquisitions, when computing metrics from real SEC filings. The dataset has 30 items with frozen evidence corpora and machine-checkable answer keys. Every item was evaluated with and without tools, first in a local harness and then as live episodes in a Gymnasium RL environment.

**Result:** comparability breaks are solved by every model in every condition. The pre-registered hypothesis is refuted, and the direction is not recommended as a training target.

- Findings and recommendation: [`report/brief.md`](report/brief.md)
- Architecture, grading, isolation, Docker and scaling: [`docs/architecture.md`](docs/architecture.md)

## Quickstart

Requires Python 3.12+ (or Docker, see below).

```bash
python -m venv venv && source venv/bin/activate
make install                  # pip install -e ".[dev]"
make test                     # 23 tests: tools, scoring, runner, judge/batch, env reward branches
```

**Reproduce every table and figure from the committed transcripts.** This needs no API keys and makes no model calls:

```bash
fh-analyze                    # Stage 1 matrix, consistency, study_heatmap_{1_iteration,3_iterations}.png
fh-gym --report               # Stage 2 matrix, returns,    gym_heatmap_{1_iteration,3_iterations}.png
```

**Re-run from scratch.** This needs API keys. `cp .env.example .env` and fill in the keys; an OpenRouter key works for GPT-5.1 and the judge.

```bash
fh-run --repeats 3 --workers 8                 # Stage 1: 540 runs, resume-safe
fh-score                                       # numeric scoring -> results/scored.csv
fh-judge --calibrate && fh-judge && fh-judge --apply   # rubric grading, gated on calibration
fh-analyze

fh-gym --repeats 3 --workers 8                 # Stage 2: 540 live episodes -> results/gym_scored.csv
```

Useful flags (shared by `fh-run` and `fh-gym`):
- `--models opus,gpt`
- `--conditions tool`
- `--items AIR-,TECH-CTRL`
- `--limit 10`

Every command also has a `make` target.

**With Docker** (no local Python needed):

```bash
make docker-test      # run the test suite inside the image
make docker-verify    # confirm the agent container cannot see any answer key or grade
make docker-run       # live Stage 1 run with API keys from .env
```
