FROM python:3.12-slim

WORKDIR /app
ENV FH_ROOT=/app PYTHONDONTWRITEBYTECODE=1

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir ".[dev]"

COPY tests/ tests/

# data/ is never baked into the image -- it's mounted at `docker run` time (see Makefile).
# This means the image itself never contains data/answer_keys.jsonl, regardless of which
# target runs it. `make docker-run` (the live agent loop) mounts only questions.jsonl +
# corpus/, so the answer key is not present anywhere in that container's filesystem --
# not "hidden," genuinely absent. `make docker-verify` proves this by listing /app/data
# inside that exact container and asserting answer_keys.jsonl isn't there.

# non-root: python_eval runs model-written code, keep it away from root
RUN useradd --create-home runner && mkdir -p transcripts/raw results && chown -R runner /app
USER runner

CMD ["pytest", "-q", "-p", "no:cacheprovider"]
