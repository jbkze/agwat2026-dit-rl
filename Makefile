.PHONY: install image

# Python 3.10 environment with every pinned dependency (CPU build of torch).
install:
	uv venv --python 3.10 .venv
	uv pip install --python .venv -e . --index-strategy unsafe-best-match \
		--extra-index-url https://download.pytorch.org/whl/cpu

# Container image, tagged with the current commit.
image:
	docker build --build-arg COMMIT=$$(git rev-parse --short HEAD) -t ditrl:$$(git rev-parse --short HEAD) .
