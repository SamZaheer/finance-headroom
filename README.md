# finance-headroom

Measures whether frontier LLMs (Claude Sonnet 5, Claude Opus 5, GPT-5.1) correctly handle **accounting comparability breaks**, such as spinoffs, discontinued-operations restatements, fiscal-year misalignment and acquisitions, when computing metrics from real SEC filings. The dataset has 30 items with frozen evidence corpora and machine-checkable answer keys. Every item was evaluated with and without tools, first in a local harness and then as live episodes in a Gymnasium RL environment.

**Result:** comparability breaks are solved by every model in every condition. The pre-registered hypothesis is refuted, and the direction is not recommended as a training target.

- Findings and recommendation: [`report/brief.md`](report/brief.md)
- Architecture, grading, isolation, Docker and scaling: [`docs/architecture.md`](docs/architecture.md)

## Run with Docker

Requires Docker and `make`. Nothing from `data/` or `.env` is baked into the image: data is mounted per command, and API keys are read from `.env` at run time.

**1. Build the image**

```bash
git clone https://github.com/SamZaheer/finance-headroom.git && cd finance-headroom
make docker-build             # docker build -t finance-headroom .
```

**2. Test** (no API keys needed)

```bash
make docker-test              # run the 23-test suite inside the image
make docker-verify            # confirm the agent container cannot see any answer key or grade
make docker-report            # rebuild every Stage 1 and Stage 2 table and figure from the committed transcripts
```

**3. Add API keys** (needed only for new model runs)

```bash
cp .env.example .env          # set ANTHROPIC_API_KEY and OPENAI_API_KEY (an OpenRouter key works for GPT-5.1 and the judge)
```

**4. Stage 1 — local evaluation harness**

```bash
make docker-run ARGS="--repeats 3 --workers 8"   # generate answers -> transcripts/raw/
make docker-score                                # numeric scoring + calibrated judge + analysis -> results/scored.csv
```

`docker-run` is the isolated agent container. It mounts only the questions and the corpus, both read-only, so the answer key and all graded outputs are absent from it. Grading runs afterwards in a separate container, `docker-score`.

**5. Stage 2 — Gymnasium environment**

```bash
make docker-gym ARGS="--repeats 3 --workers 8"   # live episodes -> gymnasium_transcripts/, results/gym_scored.csv
```

Each episode runs through `env.reset()` and `env.step()`, and the env's verifier assigns the reward inside the episode. The container therefore mounts the full `data/` directory, answer key included. See [Isolation and reward hacking](docs/architecture.md#isolation-and-reward-hacking) for this trade-off.

**Narrowing a run.** Any of these flags can go in `ARGS` for `docker-run` and `docker-gym`. Runs are resume-safe: existing transcripts are skipped.

```bash
make docker-run ARGS="--models opus --conditions tool --items TECH-CTRL --limit 5"
make docker-gym ARGS="--models claude,gpt --items AIR- --repeats 1"
```

## Run with local Python

Requires Python 3.12+.

```bash
python -m venv venv && source venv/bin/activate
make install                  # pip install -e ".[dev]"
make test
```

**Reproduce every table and figure** (no API keys, no model calls):

```bash
fh-analyze                    # Stage 1 matrix, consistency, study_heatmap_{1_iteration,3_iterations}.png
fh-gym --report               # Stage 2 matrix, returns,    gym_heatmap_{1_iteration,3_iterations}.png
```

**Re-run from scratch** (needs `.env`):

```bash
fh-run --repeats 3 --workers 8                         # Stage 1: 540 runs
fh-score && fh-judge --calibrate && fh-judge && fh-judge --apply && fh-analyze

fh-gym --repeats 3 --workers 8                         # Stage 2: 540 live episodes
```

`fh-run` and `fh-gym` accept the same flags as the Docker targets: `--models`, `--conditions`, `--items`, `--repeats`, `--workers` and `--limit`.
