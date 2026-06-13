# SIE — Superlinked Inference Engine

## What is SIE

SIE is an open-source inference engine that serves embeddings, reranking, and entity extraction through a single unified API. It replaces the patchwork of separate model servers with one system that handles 85+ models across dense, sparse, multi-vector, vision, and cross-encoder architectures.

## Requirements

- Python 3.12 or newer (see `pyproject.toml`).
- A virtual environment is recommended.

Dependencies are declared in `pyproject.toml` (e.g. `sie-sdk`, `sie-server`, `tqdm`, and Jupyter helpers).

## Setup

Install uv in the environment
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

And run the command to sync
```
uv sync
```

## Starting the SIE server 

Deploying using Docker in CPU mode
```bash
docker run -p 8080:8080 ghcr.io/superlinked/sie-server:latest-cpu-default
```
Check the documentation for other environments - https://github.com/superlinked/sie#quickstart

or start the server using sie server command
```
sie-server serve --host 0.0.0.0 --port 8080
```

## Running the scripts

From the repository root, run the scripts with Python. Examples:

- Generate embeddings using SIE:

	```bash
	python sie_extract.py
	```

- Alternative embedding script (uses sentence-transformers):

	```bash
	python sentence_transformers_embeddings.py
	```
