#!/bin/sh
# Agent run with Qwen3.8-27B (served by vLLM on the SAME A100-80GB) as the weight-editing agent.
# Memory plan on 80 GB: vLLM gets VLLM_UTIL of the card (bf16 weights ~56 GB + KV); the rest holds the agent's live
# 0.8B model and the benchmark subprocess (~6 GB at batch 32). Override via env, e.g.
#   MODEL=Qwen/Qwen3.8-27B-FP8 VLLM_UTIL=0.55 sh run_qwen38.sh      # FP8 weights ~28 GB: lots of headroom
# Backup / comparison run with OpenAI Astra:  see run_astra.sh (unchanged, uses --backend responses).
cd /workspace/wsp
export HF_HOME=/home/.cache/huggingface UV_CACHE_DIR=/home/.cache/uv
export LANGFUSE_TRACING_ENVIRONMENT=${LANGFUSE_TRACING_ENVIRONMENT:-a100-gpu} MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=${MODEL:-Qwen/Qwen3.8-27B}
VLLM_UTIL=${VLLM_UTIL:-0.82}
MAX_LEN=${MAX_LEN:-90000}
CTX_LIMIT=${CTX_LIMIT:-80000}
N=${N:-2500}                      # full test split; the agent only sees even-indexed samples, odd ones are the unseen test half
ITERATIONS=${ITERATIONS:-5}
LOG=agent_run_qwen38.log

# 1) vLLM server (own venv). Text-only agent: skip the vision tower's multimodal budget. Reuse a server that is already up.
curl -sf http://127.0.0.1:8000/v1/models > /dev/null 2>&1 || nohup /workspace/vllm-env/bin/vllm serve "$MODEL" --served-model-name qwen3.8-27b --host 127.0.0.1 --port 8000 \
  --max-model-len "$MAX_LEN" --max-num-seqs 2 --gpu-memory-utilization "$VLLM_UTIL" \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder > vllm_server.log 2>&1 &

# wait for the server (first start downloads ~56 GB), up to 40 minutes
i=0
until curl -sf http://127.0.0.1:8000/v1/models > /dev/null 2>&1; do
  i=$((i+1)); [ $i -gt 480 ] && { echo "vLLM did not come up, see vllm_server.log" > $LOG; echo "AGENT_EXIT 2" >> $LOG; exit 2; }
  sleep 5
done

# 2) mlflow
nohup uv run --no-sync mlflow server --host 127.0.0.1 --port 5000 > mlflow_server.log 2>&1 &
sleep 25

# 3) pre-flight: structured tool call + reasoning round trip must work before we spend an agent run
uv run --no-sync python llm_smoke_test.py --base-url http://127.0.0.1:8000/v1 --model qwen3.8-27b > smoke_test.log 2>&1 \
  || { cp smoke_test.log $LOG; echo "AGENT_EXIT 3 (smoke test failed)" >> $LOG; exit 3; }

# 4) the agent
uv run --no-sync python agent.py --backend chat --base-url http://127.0.0.1:8000/v1 --model qwen3.8-27b \
  --ctx-limit "$CTX_LIMIT" --iterations "$ITERATIONS" --n "$N" --batch-size 32 --probe-batch-size 8 > $LOG 2>&1
echo "AGENT_EXIT $?" >> $LOG
