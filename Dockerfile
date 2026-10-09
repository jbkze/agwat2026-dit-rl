# Image for running a shard of the plan anywhere, e.g. as a CPU-only job on a shared server.
#   make image          (builds the committed code: git archive HEAD | docker build -)
#   docker run --rm -v "$PWD/runs:/runs" ditrl:<commit> --runs /runs run smoke --workers 4
FROM python:3.10.15-slim
# git: pip installs the AquaCrop fork from GitHub.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
# Dependencies first, with only the package init, so that a code change does not reinstall them.
COPY pyproject.toml ./
COPY src/ditrl/__init__.py ./src/ditrl/
RUN pip install --no-cache-dir -e . --extra-index-url https://download.pytorch.org/whl/cpu
COPY src ./src
COPY config/plan.yaml ./config/
COPY data/nebraska_weather.csv ./data/
ARG COMMIT=unknown
# The container may run as any non-root user, so everything it writes goes to /tmp or to the
# mounted runs folder.
ENV DITRL_COMMIT=${COMMIT} PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp NUMBA_CACHE_DIR=/tmp/numba MPLCONFIGDIR=/tmp/mpl
RUN chmod -R a+rwX /work
ENTRYPOINT ["ditrl"]
