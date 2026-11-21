#!/bin/sh
# One-off: install vLLM into its OWN virtual environment (/workspace/vllm-env), separate from the project's .venv,
# because vLLM pins its own torch and would otherwise break the torch/torchvision pairing used by benchmark.py.
set -e
export HF_HOME=/home/.cache/huggingface UV_CACHE_DIR=/home/.cache/uv UV_LINK_MODE=copy
cd /workspace
uv venv vllm-env --python 3.12
# --torch-backend=auto picks the torch build that matches the instance's CUDA driver
uv pip install --python vllm-env/bin/python vllm --torch-backend=auto
vllm-env/bin/python -c "import vllm; print('vllm', vllm.__version__)"
echo VLLM_SETUP_DONE
