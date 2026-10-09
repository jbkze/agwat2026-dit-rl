# Image for a CPU-only job on the WSE vllm-job-queue (no GPU, no model manager).
#   docker build --build-arg COMMIT=$(git rev-parse --short HEAD) -t ditrl:local .
#   docker run --rm -v "$PWD/tmp-results:/results" ditrl:local \
#       config/experiments/a0_pilot_smoke.yaml --workers 4 --runs-dir /results
FROM python:3.10.15-slim
# git: pip installs the AquaCrop fork from GitHub.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e . --extra-index-url https://download.pytorch.org/whl/cpu
COPY config ./config
COPY scripts ./scripts
COPY data/nebraska_weather.csv ./data/
ARG COMMIT=unknown
# One math thread per process: a job may hold at most 512 processes and threads. HOME and the
# numba cache point to /tmp because the container may run as a non-root user.
ENV DITRL_COMMIT=${COMMIT} PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    DEVELOPMENT=DEVELOPMENT HOME=/tmp NUMBA_CACHE_DIR=/tmp/numba MPLCONFIGDIR=/tmp/mpl
RUN chmod -R a+rwX /work
ENTRYPOINT ["python", "scripts/run.py"]
