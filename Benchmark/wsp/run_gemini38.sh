#!/bin/sh
# Agent run with Google gemini-3.8-flash (OpenAI-compatible endpoint, chat backend) as the weight-editing agent.
# Same probe/benchmark setup as run_astra.sh, so the agent models are directly comparable.
cd /workspace/wsp
export PATH=$HOME/.local/bin:$PATH HF_HOME=/home/.cache/huggingface UV_CACHE_DIR=/home/.cache/uv
export LANGFUSE_TRACING_ENVIRONMENT=${LANGFUSE_TRACING_ENVIRONMENT:-l4-gpu} MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
N=${N:-2500}
ITERATIONS=${ITERATIONS:-5}
LOG=agent_run_gemini38.log
nohup uv run --no-sync mlflow server --host 127.0.0.1 --port 5000 > mlflow_server.log 2>&1 &
sleep 25
uv run --no-sync python agent.py --backend chat --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
  --api-key-env GOOGLE_API_KEY --model gemini-3.8-flash --effort high --ctx-limit 400000 \
  --iterations "$ITERATIONS" --n "$N" --batch-size 32 --probe-batch-size 8 > $LOG 2>&1
echo "AGENT_EXIT $?" >> $LOG
