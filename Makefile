.PHONY: install image

COMMIT := $(shell git rev-parse --short HEAD)

# Python 3.10 environment with every pinned dependency (CPU build of torch).
install:
	uv venv --python 3.10 .venv
	uv pip install --python .venv -e . --index-strategy unsafe-best-match \
		--extra-index-url https://download.pytorch.org/whl/cpu

# Container image of exactly the committed code (uncommitted changes are not included).
image:
	git archive --format=tar HEAD | docker build --build-arg COMMIT=$(COMMIT) -t ditrl:$(COMMIT) -
