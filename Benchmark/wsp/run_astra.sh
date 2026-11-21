#!/bin/sh
# Backup / comparison run with OpenAI gpt-6-astra (Responses API, the default backend of agent.py).
# Same probe/benchmark setup as run_qwen38.sh, so the two agent models are directly comparable.
# agent_openai_responses.py is a frozen copy of the exact agent.py that produced the earlier Astra results.
cd /workspace/wsp
export HF_HOME=/home/.cache/huggingface UV_CACHE_DIR=/home/.cache/uv
export LANGFUSE_TRACING_ENVIRONMENT=${LANGFUSE_TRACING_ENVIRONMENT:-l4-gpu} MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
N=${N:-2500}
ITERATIONS=${ITERATIONS:-5}
nohup uv run --no-sync mlflow server --host 127.0.0.1 --port 5000 > mlflow_server.log 2>&1 &
sleep 25
uv run --no-sync python agent.py --model gpt-6-astra --iterations "$ITERATIONS" --n "$N" --batch-size 32 > agent_run_astra.log 2>&1
echo "AGENT_EXIT $?" >> agent_run_astra.log
