IMAGE := finance-headroom
VOLUMES := -v $(CURDIR)/transcripts:/app/transcripts -v $(CURDIR)/results:/app/results
GYM_VOLUMES := -v $(CURDIR)/gymnasium_transcripts:/app/gymnasium_transcripts -v $(CURDIR)/results:/app/results

# Full data mount (includes answer_keys.jsonl) -- only for scoring/replay/tests, which
# read the answer key deterministically and never execute freshly-generated agent code.
DATA_FULL := -v $(CURDIR)/data:/app/data

# Agent-facing mounts -- questions + corpus read-only, and a fresh results directory.
# Neither answer_keys.jsonl nor any graded output (scored.csv, grading_draft.csv, judge_*),
# all of which contain expected answers, exists at any path inside this container.
DATA_AGENT := -v $(CURDIR)/data/questions.jsonl:/app/data/questions.jsonl:ro \
              -v $(CURDIR)/data/corpus:/app/data/corpus:ro
AGENT_VOLUMES := -v $(CURDIR)/transcripts:/app/transcripts -v $(CURDIR)/results/agent_run:/app/results

.PHONY: install test lint format run score analyze replay demo judge collect gym docker-build docker-test docker-report docker-replay docker-run docker-score docker-gym docker-verify

install:        ## editable install with dev tools
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format . && ruff check --fix .

run:            ## full model run -- needs API keys, skips existing transcripts
	fh-run $(ARGS)

score:
	fh-score

analyze:
	fh-analyze

replay:         ## replay all transcripts through the Gymnasium env
	fh-replay

demo:
	fh-demo

judge:          ## grade ungraded judgment rows with the calibrated LLM judge, then apply
	fh-judge $(ARGS) && fh-judge --apply

collect:        ## collect finished Batch API results
	fh-run --collect

gym:            ## live Gymnasium episodes: the env drives real LLM calls -> gymnasium_transcripts/
	fh-gym $(ARGS)

docker-build:
	docker build -t $(IMAGE) .

docker-test: docker-build
	docker run --rm $(DATA_FULL) $(IMAGE)

docker-report: docker-build    ## offline, no API keys: rebuild Stage 1 + Stage 2 tables and figures from committed transcripts
	docker run --rm $(DATA_FULL) $(VOLUMES) -v $(CURDIR)/gymnasium_transcripts:/app/gymnasium_transcripts \
		$(IMAGE) sh -c "fh-analyze && fh-gym --report"

docker-replay: docker-build
	docker run --rm $(DATA_FULL) $(VOLUMES) $(IMAGE) fh-score
	docker run --rm $(DATA_FULL) $(VOLUMES) $(IMAGE) fh-replay

docker-run: docker-build       ## the live agent loop -- no answer key or graded output in this container
	mkdir -p results/agent_run
	docker run --rm $(DATA_AGENT) $(AGENT_VOLUMES) --env-file .env $(IMAGE) fh-run $(ARGS)

docker-score: docker-build     ## Stage 1 grading after docker-run: numeric scoring, calibrated judge, analysis
	docker run --rm $(DATA_FULL) $(VOLUMES) --env-file .env $(IMAGE) \
		sh -c "fh-score && fh-judge && fh-judge --apply && fh-analyze"

# Stage 2 needs the full data mount: the env's verifier scores each answer inside the episode.
# python_eval therefore shares a container with the answer key (see docs/architecture.md).
docker-gym: docker-build       ## Stage 2: live Gymnasium episodes -> gymnasium_transcripts/, results/gym_scored.csv
	docker run --rm $(DATA_FULL) $(GYM_VOLUMES) --env-file .env $(IMAGE) fh-gym $(ARGS)

docker-verify: docker-build    ## proof, not a claim: list what the agent's own container can see
	mkdir -p results/agent_run
	@echo "Files visible to the agent container under /app/data and /app/results:"
	@docker run --rm $(DATA_AGENT) $(AGENT_VOLUMES) $(IMAGE) find /app/data /app/results -type f
	@echo "Asserting no answer key or graded output is present..."
	@! docker run --rm $(DATA_AGENT) $(AGENT_VOLUMES) $(IMAGE) \
		find /app -name 'answer_keys.jsonl' -o -name 'scored.csv' -o -name 'gym_scored.csv' \
		-o -name 'grading_draft.csv' -o -name 'judge_*' | grep -q .
	@echo "PASS: no answer key or graded output is present in the agent container."
