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
| `agent.py` | The agent (OpenAI `gpt-6-luna` via the Responses API). Tools: `python` (sandboxed weight inspection/edit on a live GPU copy of the model), `probe` (free, cheap measurement), `run_benchmark`, `revert`, `finish` |

### Agent model backends

`agent.py --backend responses` (default) drives OpenAI models (`gpt-6-luna`, `gpt-6-astra`) through the Responses API.
`agent.py --backend chat --base-url <url> --model <name>` drives any OpenAI-compatible Chat Completions server, e.g.
**Qwen3.8-27B** served by vLLM on an A100-80GB. The chat backend keeps the conversation locally, sends the model's
reasoning back each turn (preserved thinking) and trims old tool output before the context window fills.

| Script | Purpose |
|---|---|
| `setup_vllm.sh` | one-off: install vLLM into its own venv (`/workspace/vllm-env`) |
| `run_qwen38.sh` | vLLM + MLflow + tool-call smoke test (`llm_smoke_test.py`) + agent run with Qwen3.8-27B |
| `run_astra.sh` | the same run with OpenAI `gpt-6-astra`, for comparison |
| `agent_openai_responses.py` | frozen copy of the original OpenAI-only agent (produced the earlier Astra results) |

Both scripts default to `--n 2000`, which leaves test samples 2000-2499 held out for the agent's probe set, so gains can be
checked on data the agent did not tune on.

### How the agent edits weights

Earlier versions edited weights blind and could only test an edit with a full benchmark run. They ended up sweeping global
scale factors, and every result stayed within noise (±0.3 points). The agent now follows the weight-space-programming loop:

1. **Error taxonomy.** Baseline errors are split into *obstacles* (the output format breaks scoring: truncated
   step-by-step reasoning, the right number with extra text, a value given for a "Which…?" question) and capability
   errors (misreadings, arithmetic), which edits can't fix.
2. **Look inside.** In the sandbox, `first_tokens()` and `capture()` read the next-token distribution and residual-stream
   activations at the answer position. `contrast_direction()` finds a behaviour direction from two groups of samples.
3. **Directional edit.** `edit_direction(d, layers, factor)` rescales what the chosen layers write along `d`
   (`W_out ← W_out + (factor−1)·d dᵀ W_out`). This is the refusal-direction-ablation style of edit.
4. **Probe.** `probe()` scores the current weights on a probe set in about a minute and reports fixed/broken samples vs
   the unedited model with a sign-test p-value. It uses held-out test samples when `--n` leaves some, otherwise
   benchmark samples enriched with obstacle cases.
5. **Benchmark.** The agent spends `run_benchmark` only on edits that probed well. Each result reports gained/lost samples
   vs the baseline and a sign test, so real gains can be told apart from noise.
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

On a VM (e.g. L4): `uv run python agent.py --model gpt-6-astra --iterations 5 --n 2500 --batch-size 32`.
Use `--n 2000` to keep 500 test samples held out for probing. With `--n 2500` the probe set overlaps the benchmark.

Probe options: `--probe-n 300` (probe set size), `--max-probes 40`, `--probe-batch-size`, `--offload` (move the agent's
live model to the CPU while the benchmark runs, for small GPUs).

Useful options: `--model gpt-6-luna` (agent model), `--effort high` (reasoning effort),
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
