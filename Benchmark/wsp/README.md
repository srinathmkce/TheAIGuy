# Weight-space programming on ChartQA

An LLM agent improves **Qwen3.5-0.8B** (a vision-language model) on **ChartQA** by reading and directly
editing the model's weights. There is no training and no gradient descent. After each edit the agent runs a
locked benchmark and uses the score and the wrong answers as feedback.

Based on *Sparks of Weight-Space Programming* (Takehi et al.) and *Weight-space programming for auditing
clinical decision networks* (arXiv 2611.05341).

## How it works

```
agent.py  --(python tool: read/edit weights)-->  edited tensors (delta)
   |                                                  |
   |<--- accuracy + wrong answers + MLflow run ---  benchmark.py (locked, runs in its own process on the GPU)
```

| File | Role |
|---|---|
| `download.py` | Downloads the model to `Qwen3.5-0.8B/` and the ChartQA **test** split to `data/ChartQA_test` |
| `benchmark.py` | The locked benchmark: load model on GPU, apply an optional weight delta, run the first N test samples, score, log to MLflow, exit |
| `agent.py` | The agent (OpenAI `gpt-6-luna` via the Responses API). Tools: `python` (sandboxed weight inspection/edit), `run_benchmark`, `revert`, `finish` |
| `results/` | Metrics and predictions per run (`agent_<timestamp>/` holds the agent's transcript, deltas and `summary.json`) |

**Metric.** ChartQA *relaxed accuracy*: numeric answers within 5% of the target, otherwise exact match
(case-insensitive). Greedy decoding, thinking off, at most 32 new tokens, bf16. The benchmark always uses the
**first N test samples**, so every run is directly comparable.

**The agent cannot change the benchmark.** It has no file or import access. Its sandbox only exposes the weights
(`W` editable, `B` read-only originals). The runner also hashes `benchmark.py`, the dataset and the base weights
after every tool call and aborts if anything changed. The original model in `Qwen3.5-0.8B/` is never modified;
edits are saved as separate delta files and applied in memory for each run.

## Setup

Requires Python 3.13, [uv](https://docs.astral.sh/uv/), an NVIDIA GPU (tested on a 4 GB RTX 3050) and an MLflow server.

```powershell
# 1. Install into the project virtual environment (torch is pinned to the CUDA 12.8 build)
uv sync

# 2. Add keys to .env (never commit this file)
#    OPENAI_API_KEY=...

# 3. Download the model and the ChartQA test split into this folder
uv run python download.py

# 4. Start the MLflow server in another terminal (default http://127.0.0.1:5000)
mlflow server
```

If the MLflow server is elsewhere, change `MLFLOW_URI` in `benchmark.py`.

## Run

```powershell
# Baseline: unedited model on the first 500 test samples. Logged to MLflow as the run "baseline".
uv run python benchmark.py --n 500 --run-name baseline --stage baseline --note "Unedited Qwen3.5-0.8B" --tag baseline_n500

# Agent: up to 5 iterations of (edit weights -> benchmark -> feedback), 500 samples each.
# It reuses results/metrics_baseline_n500.json from the step above instead of re-running the baseline.
uv run python agent.py --iterations 5 --n 500
```

Useful options: `--model gpt-6-luna` (agent model), `--effort medium` (reasoning effort),
`--price-in/--price-out` (USD per 1M tokens, for a manual cost estimate), `--max-turns 120`.

Set `PYTHONUTF8=1` if your console is not UTF-8 (MLflow prints emoji).

## Results in MLflow

Everything is logged to the experiment `ChartQA-WeightSpaceProgramming`:

- **`baseline`** and **`agent_<timestamp>`** (parent run) with nested runs **`iter1` ... `iterN`**. Compare them in the
  UI (select runs -> Compare) on `relaxed_accuracy`, `n_correct`, timings and GPU memory.
- Each iteration run has the agent's `note` (what it changed and why), the number of edited tensors, the predictions
  and the weight delta as artifacts.
- The parent run has `baseline_accuracy`, `best_accuracy`, `improvement`, the agent summary, and token usage.
- **Traces** tab: every agent LLM call and tool call (`agent_session` trace) with token usage and cost
  (`agent_trace_cost_usd` on the parent run).

## Notes and limitations

- Qwen3.5-0.8B with thinking off does not loop, so there is less to remove than in the paper's math experiment, which
  fixed looping. Weight edits can remove an obstacle but are poor at adding capability.
- At N=500 one answer is 0.2 percentage points. Differences of about a point are within noise.
- The first N samples come from one subset of ChartQA (`human_or_machine` = 0), so scores reflect that subset only.
- The machine's Windows page file is small. `agent.py` loads weights lazily and `benchmark.py` loads the model
  straight onto the GPU so both processes fit in memory together. If you see "The paging file is too small",
  close other apps or enlarge the page file.
