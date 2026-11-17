"""Weight-space-programming agent for Qwen3.5-0.8B on ChartQA.

An LLM (default: OpenAI gpt-6-luna) inspects and edits the Qwen weights through a sandboxed
Python tool, then calls `run_benchmark` (the locked benchmark.py, run in a subprocess on the GPU)
and iterates on the feedback. No gradient training: edits are written directly into the tensors.

The agent can never touch the benchmark: it only gets the tools below, its python sandbox has no
file/import access, and the runner verifies benchmark.py + dataset + base weights are unchanged
after every tool call.

Usage: uv run python agent.py --iterations 10
"""
import argparse, contextlib, hashlib, io, json, math, os, random, re, subprocess, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from dotenv import dotenv_values
from openai import OpenAI
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "Qwen3.5-0.8B"
BENCH = ROOT / "benchmark.py"
N_SAMPLES = 500

SYSTEM = """You are a weight-space programmer. You improve a small multimodal model by reading and directly editing its weights.

Target: Qwen3.5-0.8B (vision-language, bf16) answering ChartQA questions (chart image + question -> short answer).
Score: relaxed accuracy (numbers within 5%, otherwise exact match, case-insensitive) on a fixed set of 500 test
samples, greedy decoding, thinking OFF, at most 32 new tokens, answer = last line of the output. The model is
told to reply with only the final answer. Baseline accuracy is reported by the first run.

Rules
- You change the model ONLY by editing tensors in the dict `W` inside the `python` tool (name -> torch parameter,
  edit in place, e.g. W[name].data[...] = ...). No training, no gradient descent, no external data. Direct edits only.
- `B` reads the untouched original tensors on demand (B[name], B.keys(), B.shape(name)). Weights are loaded lazily:
  iterate names with W.keys()/W.shape(name); for bulk scans over many tensors use W.peek(name) or W.items() (read-only copies,
  edits to them are NOT kept); to EDIT a tensor, access it as W[name] and modify it in place, or assign W[name] = new_tensor.
  Memory is tight, so avoid holding many large tensors at once. `torch`, `np`, `math`, `re` are available.
  There is no file, network or import access, and no access to the benchmark or dataset.
- `run_benchmark(note)` saves your current edits, evaluates them on the 500 samples and returns accuracy, how many
  tensors/parameters differ from the base, and a sample of wrong answers (query, target, prediction).
  Each call uses one iteration of a limited budget; the python tool is free. Think and analyse before spending one.
- `revert(to)` restores the best-scoring edits ('best') or the original weights ('base').
- Call `finish(summary)` when you are done or have no promising ideas left.

Guidance (from the weight-space-programming work)
- Edits are good at removing an obstacle (a harmful direction, a bad bias, a miscalibrated scale/norm, a formatting habit
  such as verbosity or units/percent signs that breaks exact match), and poor at adding capabilities the model lacks.
- Small, targeted, interpretable edits (a few rows/columns of a few layers) beat broad perturbations. Always measure
  what you changed (norm of delta, number of parameters) and hypothesise what behaviour it should change and why.
- Use the wrong-answer samples to find systematic failure modes (e.g. answers given with extra words, wrong format,
  rounding, a missing %), then look in the weights for where that behaviour is produced.
- A run that does not improve is information: revert and try a different hypothesis. Do not randomly perturb weights.
"""

TOOLS = [
    {"type": "function", "name": "python", "description":
        "Run Python in a sandbox to inspect/edit weights. Variables: W (editable params dict), B (original params), torch, np, math, re. "
        "Use print() to see output (truncated to ~6000 chars). State persists between calls through W only.",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}},
    {"type": "function", "name": "run_benchmark", "description":
        "Evaluate the current edited weights on the 500-sample ChartQA benchmark (uses 1 iteration, takes a few minutes).",
     "parameters": {"type": "object", "properties": {"note": {"type": "string", "description": "what you changed and why"}}, "required": ["note"]}},
    {"type": "function", "name": "revert", "description": "Restore the best-scoring edits or the original weights.",
     "parameters": {"type": "object", "properties": {"to": {"type": "string", "enum": ["best", "base"]}}, "required": ["to"]}},
    {"type": "function", "name": "finish", "description": "Stop and report.",
     "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
]

SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n) for n in
                 "abs all any bool dict enumerate float int len list max min print range round set sorted sum tuple zip isinstance reversed slice str map filter divmod pow repr type".split()}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint():
    """Hash of everything the agent must not change: benchmark code, dataset, base weights."""
    files = [BENCH, *sorted((ROOT / "data").rglob("*.arrow")), *sorted(MODEL_DIR.glob("*.safetensors*"))]
    return {str(f.relative_to(ROOT)): sha(f) for f in files}


class TensorFile:
    """Reads single tensors from a safetensors file with plain file I/O (no mmap: a mapped 1.7GB file is
    charged against Windows' commit limit, which is already tight on this machine)."""
    DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}

    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            self.meta = {k: v for k, v in json.loads(f.read(n)).items() if k != "__metadata__"}
        self.base = 8 + n

    def keys(self): return list(self.meta)
    def shape(self, k): return tuple(self.meta[k]["shape"])
    def __contains__(self, k): return k in self.meta
    def __iter__(self): return iter(self.meta)
    def __len__(self): return len(self.meta)

    def __getitem__(self, k):
        m = self.meta[k]; s, e = m["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.base + s); buf = bytearray(f.read(e - s))
        return torch.frombuffer(buf, dtype=self.DT[m["dtype"]]).reshape(m["shape"])


class LazyWeights:
    """Dict-like editable weights. A tensor is read from disk on first access (W[name]) and kept; tensors
    that differ from the base are kept permanently, clean ones are dropped between python tool calls so
    memory stays proportional to what the agent actually edits."""
    def __init__(self, base):
        self.base, self.live, self.size = base, {}, 0

    def keys(self): return self.base.keys()
    def shape(self, k): return self.base.shape(k)
    def __contains__(self, k): return k in self.base
    def __iter__(self): return iter(self.base)
    def __len__(self): return len(self.base)
    def __getitem__(self, k):
        if k not in self.live:
            self.live[k] = self.base[k]
        return self.live[k]
    def __setitem__(self, k, v):
        if tuple(v.shape) != self.base.shape(k):
            raise ValueError(f"shape mismatch for {k}")
        self.live[k] = v.to(self.base[k].dtype).contiguous()
    def peek(self, k):
        """Read-only copy for bulk scans (norms etc.). Edits to it are NOT kept; use W[name] to edit."""
        return self.live[k] if k in self.live else self.base[k]
    def items(self):
        for k in self.keys():
            yield k, self.peek(k)
    def values(self):
        for k in self.keys():
            yield self.peek(k)
    def compact(self):
        """Drop clean tensors (call between tool calls, never while the agent holds references)."""
        with torch.no_grad():
            for k in [k for k, t in self.live.items() if torch.equal(t, self.base[k])]:
                del self.live[k]

    def delta(self):
        self.compact()
        return {k: t.detach().clone().contiguous() for k, t in self.live.items()}

    def restore(self, state):
        self.live = {k: t.clone() for k, t in (state or {}).items()}


class Workspace:
    def __init__(self):
        cache = ROOT / "cache" / "base_params.safetensors"
        if not cache.exists():  # one-off: parameter names must match the module names the benchmark uses
            model = AutoModelForImageTextToText.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
            cache.parent.mkdir(exist_ok=True)
            save_file({k: v.detach().contiguous() for k, v in model.named_parameters()}, str(cache))
            del model
        self.B = TensorFile(cache)
        self.W = LazyWeights(self.B)
        self.best_state = None  # delta of best run: name -> tensor

    def exec(self, code):
        self.W.compact()
        if "__" in code or re.search(r"\b(import|open|exec|eval|compile|globals|locals|getattr|setattr|delattr|input|breakpoint)\b", code):
            return "ERROR: forbidden construct (import/open/exec/eval/dunder/getattr...). Use only W, B, torch, np, math, re."
        buf = io.StringIO()
        ns = {"__builtins__": SAFE_BUILTINS, "W": self.W, "B": self.B, "torch": torch, "np": np, "math": math, "re": re}
        try:
            with contextlib.redirect_stdout(buf), torch.no_grad():
                exec(code, ns)
        except Exception as e:  # report to the agent
            buf.write(f"\nERROR: {type(e).__name__}: {e}")
        out = buf.getvalue()
        return out[:6000] + ("\n...[truncated]" if len(out) > 6000 else "") or "(no output)"

    def delta(self): return self.W.delta()
    def restore(self, state): self.W.restore(state)


def run_bench(delta_path, out_dir, tag, n=N_SAMPLES, run_name=None, stage="agent_iteration", parent=None, note="", iteration=0):
    cmd = [sys.executable, str(BENCH), "--n", str(n), "--out-dir", str(out_dir), "--tag", tag]
    cmd += ["--run-name", run_name or tag, "--stage", stage, "--note", note, "--iteration", str(iteration)]
    if parent:
        cmd += ["--parent-run-id", parent]
    if delta_path:
        cmd += ["--delta", str(delta_path)]
    mfile = Path(out_dir) / f"metrics_{tag}.json"
    for attempt in range(3):  # free RAM/VRAM on this shared machine fluctuates: retry with a pause
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, encoding="utf-8", errors="replace")
        if r.returncode == 0 and mfile.exists():
            break
        time.sleep(30)
    else:
        return None, None, r.stderr[-1500:]
    return json.loads(mfile.read_text()), json.loads((Path(out_dir) / f"predictions_{tag}.json").read_text()), ""


def summarize_run(metrics, preds, delta, base_acc, k=12):
    wrong = [p for p in preds if not p["correct"]]
    sample = random.Random(len(preds)).sample(wrong, min(k, len(wrong)))
    return {
        "relaxed_accuracy": metrics["relaxed_accuracy"], "delta_vs_baseline": round(metrics["relaxed_accuracy"] - base_acc, 4),
        "edited_tensors": len(delta),
        "wrong_answer_samples": [{"query": w["query"], "target": w["target"], "prediction": w["prediction"]} for w in sample],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=10, help="benchmark runs the agent may spend")
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max-turns", type=int, default=120)
    ap.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens, to log estimated agent cost")
    ap.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens (incl. reasoning)")
    ap.add_argument("--n", type=int, default=N_SAMPLES, help="samples per run (use a small value for smoke tests)")
    args = ap.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "results" / f"agent_{run_id}"
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "transcript.jsonl", "a", encoding="utf-8")
    def L(**kw): log.write(json.dumps({"t": time.time(), **kw}, default=str) + "\n"); log.flush()

    import mlflow
    from benchmark import MLFLOW_URI, MLFLOW_EXPERIMENT
    mlflow.set_tracking_uri(MLFLOW_URI); mlflow.set_experiment(MLFLOW_EXPERIMENT)
    mlflow.openai.autolog()  # traces every LLM call (prompts, tool calls, reasoning/usage) in the MLflow Traces tab
    parent_run = mlflow.start_run(run_name=f"agent_{run_id}", tags={"stage": "agent_session", "agent_model": args.model})
    parent = parent_run.info.run_id
    mlflow.log_params({"agent_model": args.model, "iterations_budget": args.iterations, "n_samples": args.n, "reasoning_effort": args.effort})

    guard = fingerprint()
    bm, bp = ROOT / "results" / f"metrics_baseline_n{args.n}.json", ROOT / "results" / f"predictions_baseline_n{args.n}.json"
    if bm.exists() and bp.exists():  # the recorded MLflow "baseline" run: reuse its results, don't re-run
        m0, p0 = json.loads(bm.read_text()), json.loads(bp.read_text())
        print("Using recorded baseline", bm.name, flush=True)
    else:
        print("Baseline run (iteration 0)...", flush=True)
        m0, p0, err = run_bench(None, out, "iter0_base", n=args.n, run_name="iter0_base", stage="agent_baseline", parent=parent)
        assert m0, f"baseline failed: {err}"
    base_acc = m0["relaxed_accuracy"]
    print(f"baseline tune accuracy = {base_acc:.3f}", flush=True)

    ws = Workspace()
    client = OpenAI(api_key=dotenv_values(ROOT / ".env")["OPENAI_API_KEY"])
    history = [{"tune_iter": 0, "note": "base", "accuracy": base_acc}]
    best = {"acc": base_acc, "iter": 0}
    used, infra_fails = 0, 0
    wrong0 = [p for p in p0 if not p["correct"]]
    first = {"relaxed_accuracy": base_acc, "wrong_answer_samples": [
        {"query": w["query"], "target": w["target"], "prediction": w["prediction"]}
        for w in random.Random(0).sample(wrong0, min(12, len(wrong0)))]}
    prompt = (f"Iteration budget: {args.iterations} benchmark runs. Baseline (unedited) result:\n"
              + json.dumps(first, indent=1) + "\nStart by inspecting the weights (names, shapes, norms).")
    inp, prev, finished, summary = [{"role": "user", "content": prompt}], None, False, ""

    usage = {"llm_calls": 0, "llm_input_tokens": 0, "llm_cached_input_tokens": 0, "llm_output_tokens": 0, "llm_reasoning_tokens": 0}
    root = mlflow.start_span(name="agent_session", span_type="AGENT")
    root_span = root.__enter__(); root_span.set_inputs({"model": args.model, "iterations": args.iterations, "n_samples": args.n, "baseline": base_acc})
    for turn in range(args.max_turns):
        if finished:
            break
        r = client.responses.create(model=args.model, instructions=SYSTEM, input=inp, tools=TOOLS,
                                    reasoning={"effort": args.effort}, previous_response_id=prev, max_output_tokens=20000)
        prev, inp = r.id, []
        u = r.usage
        usage["llm_calls"] += 1
        usage["llm_input_tokens"] += u.input_tokens
        usage["llm_cached_input_tokens"] += u.input_tokens_details.cached_tokens
        usage["llm_output_tokens"] += u.output_tokens
        usage["llm_reasoning_tokens"] += u.output_tokens_details.reasoning_tokens
        calls = [o for o in r.output if o.type == "function_call"]
        for o in r.output:
            if o.type == "message":
                text = "".join(c.text for c in o.content if c.type == "output_text")
                L(kind="assistant", text=text); print(f"[agent] {text[:400]}", flush=True)
        if not calls:
            inp = [{"role": "user", "content": "Continue: call a tool (python / run_benchmark / revert / finish)."}]
            continue
        for c in calls:
            args_j = json.loads(c.arguments or "{}")
            L(kind="call", name=c.name, args=args_j)
            tool_cm = mlflow.start_span(name=f"tool:{c.name}", span_type="TOOL")
            tool_span = tool_cm.__enter__(); tool_span.set_inputs(args_j)
            if c.name == "python":
                res = ws.exec(args_j.get("code", ""))
            elif c.name == "revert":
                ws.restore(ws.best_state if args_j.get("to") == "best" else None)
                res = f"reverted to {args_j.get('to')}"
            elif c.name == "finish":
                finished, summary, res = True, args_j.get("summary", ""), "ok"
            elif c.name == "run_benchmark":
                if used >= args.iterations:
                    res = "ERROR: iteration budget exhausted. Call finish."
                else:
                    used += 1
                    delta = ws.delta()
                    dpath = out / f"iter{used}_delta.safetensors"
                    save_file(delta, str(dpath))
                    n_par = sum(t.numel() for t in delta.values())
                    print(f"[iter {used}/{args.iterations}] running benchmark: {len(delta)} tensors changed -- {args_j.get('note','')[:150]}", flush=True)
                    m, p, err = run_bench(dpath if delta else None, out, f"iter{used}", n=args.n,
                                    run_name=f"iter{used}", stage="agent_iteration", parent=parent, note=args_j.get("note", ""), iteration=used)
                    if m is None:  # infrastructure failure, not a result: refund the iteration
                        used -= 1; infra_fails += 1
                        res = ("INFRASTRUCTURE ERROR (iteration refunded, not caused by your edit; just call run_benchmark again): "
                               + err[-600:])
                        if infra_fails >= 3:
                            raise RuntimeError(f"benchmark infrastructure failed {infra_fails} times: {err[-500:]}")
                    else:
                        s = summarize_run(m, p, delta, base_acc)
                        s["edited_parameters"] = n_par
                        s["iterations_left"] = args.iterations - used
                        history.append({"tune_iter": used, "note": args_j.get("note", ""), "accuracy": m["relaxed_accuracy"],
                                        "edited_tensors": len(delta), "edited_parameters": n_par})
                        if m["relaxed_accuracy"] > best["acc"]:
                            best, ws.best_state = {"acc": m["relaxed_accuracy"], "iter": used}, delta
                            s["new_best"] = True
                        print(f"[iter {used}] accuracy {m['relaxed_accuracy']:.3f} (base {base_acc:.3f}, best {best['acc']:.3f})", flush=True)
                        res = json.dumps(s, indent=1)
            else:
                res = f"unknown tool {c.name}"
            L(kind="result", name=c.name, result=res[:4000])
            tool_span.set_outputs(res[:4000])
            tool_cm.__exit__(None, None, None)
            inp.append({"type": "function_call_output", "call_id": c.call_id, "output": res})
            if fingerprint() != guard:
                raise RuntimeError("Integrity check failed: benchmark/dataset/base weights changed. Aborting.")

    root.__exit__(None, None, None)
    # No held-out set: every run (baseline and iterations) uses the same 500 samples, so scores are directly comparable.
    if ws.best_state:
        save_file(ws.best_state, str(out / "best_delta.safetensors"))
    del ws
    final = {
        "run_id": run_id, "agent_model": args.model, "n_samples": args.n, "iterations_used": used,
        "baseline_accuracy": base_acc, "best_accuracy": best["acc"], "best_iteration": best["iter"],
        "improvement": round(best["acc"] - base_acc, 4),
        "history": history, "agent_summary": summary,
    }
    (out / "summary.json").write_text(json.dumps(final, indent=2))
    mlflow.log_metrics({"baseline_accuracy": base_acc, "best_accuracy": best["acc"], "improvement": final["improvement"],
                        "iterations_used": used})
    mlflow.set_tags({"best_iteration": str(best["iter"])})
    cost = {}
    if args.price_in is not None and args.price_out is not None:
        cost["agent_est_cost_usd"] = (usage["llm_input_tokens"] - usage["llm_cached_input_tokens"]) * args.price_in / 1e6             + usage["llm_output_tokens"] * args.price_out / 1e6
    try:  # MLflow computes cost per trace from its model price table
        for _ in range(10):
            try:
                tm = mlflow.get_trace(root_span.trace_id).info.trace_metadata
                if "mlflow.trace.cost" in tm:
                    cost["agent_trace_cost_usd"] = json.loads(tm["mlflow.trace.cost"])["total_cost"]; break
            except Exception:
                pass
            time.sleep(1)
    except Exception as e:
        print("trace cost unavailable:", e)
    mlflow.log_metrics({**usage, **cost})
    final["agent_usage"] = {**usage, **cost}
    (out / "summary.json").write_text(json.dumps(final, indent=2))
    mlflow.log_artifact(str(out / "summary.json")); mlflow.log_artifact(str(out / "transcript.jsonl"))
    mlflow.end_run()
    print(json.dumps({k: v for k, v in final.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
