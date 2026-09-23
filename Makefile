IMAGE := finance-headroom
# local targets use the project's venv when it exists, so `make` works whether or not a venv is
# activated (make's /bin/sh only finds fh-* if venv/bin is on PATH)
BIN := $(if $(wildcard $(CURDIR)/venv/bin/fh-run),$(CURDIR)/venv/bin/,)
VOLUMES := -v $(CURDIR)/transcripts:/app/transcripts -v $(CURDIR)/results:/app/results
# Stage 2 sees ONLY its own outputs: results/gym, never Stage 1's results/ (scored.csv, judge files, figures)
GYM_VOLUMES := -v $(CURDIR)/gymnasium_transcripts:/app/gymnasium_transcripts -v $(CURDIR)/results/gym:/app/results/gym

# Full data mount (includes answer_keys.jsonl) -- only for scoring/replay/tests, which
# read the answer key deterministically and never execute freshly-generated agent code.
DATA_FULL := -v $(CURDIR)/data:/app/data

# Agent-facing mounts -- questions + corpus read-only, and a fresh results directory.
# Neither answer_keys.jsonl nor any graded output (scored.csv, grading_draft.csv, judge_*),
# all of which contain expected answers, exists at any path inside this container.
DATA_AGENT := -v $(CURDIR)/data/questions.jsonl:/app/data/questions.jsonl:ro \
              -v $(CURDIR)/data/corpus:/app/data/corpus:ro
AGENT_VOLUMES := -v $(CURDIR)/transcripts:/app/transcripts -v $(CURDIR)/results/agent_run:/app/results

.PHONY: dirs clean docker-fresh gym-draft-gold gym-gold docker-gym-draft-gold docker-gym-gold install test lint format run score analyze replay demo judge collect gym gold calibrate docker-build docker-test docker-report docker-replay docker-run docker-score docker-gym docker-gym-score docker-verify

install:        ## editable install with dev tools
	$(BIN)pip install -e ".[dev]"

test:
	$(BIN)pytest -q

lint:
	$(BIN)ruff check .

format:
	$(BIN)ruff format . && $(BIN)ruff check --fix .

run: dirs            ## full model run -- needs API keys, skips existing transcripts
	$(BIN)fh-run $(ARGS)

score: dirs
	$(BIN)fh-score

analyze:
	$(BIN)fh-analyze

replay:         ## replay all transcripts through the Gymnasium env
	$(BIN)fh-replay

demo:
	$(BIN)fh-demo

judge:          ## grade ungraded judgment rows with the calibrated LLM judge, then apply
	$(BIN)fh-judge $(ARGS) && $(BIN)fh-judge --apply

collect:        ## collect finished Batch API results
	$(BIN)fh-run --collect

gold: dirs           ## ONE-TIME: freeze approved judgment grades (+ answers) into data/calibration/judge_gold.jsonl
	$(BIN)fh-judge --merge-approved && $(BIN)fh-judge --freeze-gold

calibrate:      ## calibrate the judge against the gold set (fh-gym also does this automatically when needed)
	$(BIN)fh-judge --calibrate

gym-draft-gold: dirs  ## Stage 2: judgment episodes + judge verdicts -> results/gym/grading_draft.csv (approve there)
	$(BIN)fh-gym --draft-gold

gym-gold: dirs        ## Stage 2: approved rows -> data/calibration/gym_gold.jsonl (Stage 2's own judge baseline)
	$(BIN)fh-gym --freeze-gold

gym: dirs            ## live Gymnasium episodes: the env drives real LLM calls -> gymnasium_transcripts/
	$(BIN)fh-gym $(ARGS)

# folders every stage reads or writes; created up front so no run ever fails on a missing
# directory (and Docker never has to create a mount point itself)
dirs:
	@mkdir -p transcripts/raw gymnasium_transcripts results/agent_run results/gym data/calibration

docker-build: dirs
	docker build -t $(IMAGE) .

# explicit clean slate -- deletes every model answer and live episode (paid API calls), so it is
# never a side effect of building; results/ and data/calibration/ (the judge's gold set) are kept
clean:
	rm -rf transcripts gymnasium_transcripts
	@echo "removed transcripts/ and gymnasium_transcripts/ (results/ and data/calibration/ kept)"

docker-fresh: clean docker-build   ## clean slate + rebuild; folders are recreated empty

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
# python_eval therefore shares a container with the answer key (see docs/architecture.md) -- unlike
# Stage 1, this step can't be isolated from the answer key, since the reward is assigned live,
# inside env.step(), not as a separate offline pass.
docker-gym: docker-build       ## Stage 2 episodes -> gymnasium_transcripts/ (reward assigned live, inside each episode)
	docker run --rm $(DATA_FULL) $(GYM_VOLUMES) --env-file .env $(IMAGE) fh-gym $(ARGS) --no-report

docker-gym-draft-gold: docker-build  ## Stage 2 gold draft from Stage 2 episodes only
	docker run --rm $(GYM_VOLUMES) $(IMAGE) fh-gym --draft-gold

docker-gym-gold: docker-build        ## freeze approved Stage 2 rows into data/calibration/gym_gold.jsonl
	docker run --rm $(GYM_VOLUMES) -v $(CURDIR)/data/calibration:/app/data/calibration $(IMAGE) fh-gym --freeze-gold

docker-gym-score: docker-build ## Stage 2 report after docker-gym: csv + heatmap from stored episodes, no answer key needed, no API calls
	docker run --rm $(GYM_VOLUMES) $(IMAGE) fh-gym --report

docker-verify: docker-build    ## proof, not a claim: list what the agent's own container can see
	mkdir -p results/agent_run
	@echo "Files visible to the agent container under /app/data and /app/results:"
	@docker run --rm $(DATA_AGENT) $(AGENT_VOLUMES) $(IMAGE) find /app/data /app/results -type f
	@echo "Asserting no answer key or graded output is present..."
	@! docker run --rm $(DATA_AGENT) $(AGENT_VOLUMES) $(IMAGE) \
		find /app -name 'answer_keys.jsonl' -o -name 'scored.csv' -o -name 'gym_scored.csv' \
		-o -name 'grading_draft.csv' -o -name 'judge_*' | grep -q .
	@echo "PASS: no answer key or graded output is present in the agent container."
