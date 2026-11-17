"""ChartQA benchmark for Qwen3.5-0.8B (multimodal: the model sees the chart image + question).

Usage: uv run python benchmark.py --n 20
"""
import argparse, json, re, sys, time
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_from_disk
import transformers.modeling_utils as _mu
from transformers import AutoModelForImageTextToText, AutoProcessor

# transformers pre-allocates one model-sized (1.6 GiB) CUDA block before loading; on a shared 4 GB laptop GPU that single
# allocation can fail when other apps (browser) hold VRAM. Skipping the warm-up loads tensors one at a time instead.
_mu.caching_allocator_warmup = lambda *a, **k: None

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # MLflow prints emoji; cp1252 consoles crash
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "Qwen3.5-0.8B"
DATA_DIR = ROOT / "data" / "ChartQA_test"
RESULTS_DIR = ROOT / "results"


def _to_float(s):
    s = s.strip().replace(",", "").replace("$", "")
    pct = s.endswith("%")
    if pct:
        s = s[:-1]
    try:
        return float(s)
    except ValueError:
        return None


def relaxed_correct(pred, target, tol=0.05):
    """Standard ChartQA relaxed accuracy: 5% tolerance on numbers, exact (case-insens.) otherwise."""
    pred, target = pred.strip().lower(), target.strip().lower()
    p, t = _to_float(pred), _to_float(target)
    if p is not None and t is not None:
        if t == 0:
            return p == 0
        return abs(p - t) / abs(t) <= tol
    return pred == target


def extract_answer(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return text.splitlines()[-1].strip().rstrip(".") if text else ""


MLFLOW_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT = "ChartQA-WeightSpaceProgramming"


def log_mlflow(args, metrics, files, records):
    """Every benchmark run is one MLflow run in the same experiment, so runs are directly comparable."""
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    name = args.run_name or ("baseline" if not args.delta else "edited")
    with mlflow.start_run(run_name=name, parent_run_id=args.parent_run_id, nested=bool(args.parent_run_id)):
        mlflow.set_tags({"stage": args.stage, "model": metrics["model"],
                         "edited": str(bool(args.delta)), "note": args.note[:5000]})
        mlflow.log_params({"n_samples": metrics["n_samples"], "batch_size": metrics["batch_size"],
                           "max_new_tokens": args.max_new_tokens, "max_pixels": args.max_pixels, "iteration": args.iteration,
                           "edited_tensors": metrics["edited_tensors"]})
        m = {"relaxed_accuracy": metrics["relaxed_accuracy"], "inference_seconds": metrics["inference_seconds"],
             "model_load_seconds": metrics["model_load_seconds"], "seconds_per_sample": metrics["seconds_per_sample"],
             "peak_gpu_mem_gb": metrics["peak_gpu_mem_gb"], "n_correct": sum(r["correct"] for r in records)}
        for k, v in metrics["accuracy_by_split"].items():
            m[f"accuracy_human_or_machine_{k}"] = v
        mlflow.log_metrics(m, step=args.iteration)
        for f in files:
            mlflow.log_artifact(str(f))
        if args.delta and Path(args.delta).stat().st_size < 200e6:
            mlflow.log_artifact(args.delta, "weight_delta")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--delta", default=None, help="safetensors file with edited tensors applied over the base weights")
    ap.add_argument("--out-dir", default=str(RESULTS_DIR))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--run-name", default=None, help="MLflow run name (e.g. baseline, iter3)")
    ap.add_argument("--stage", default="baseline", help="MLflow tag: baseline | agent_iteration")
    ap.add_argument("--parent-run-id", default=None, help="MLflow parent run (agent session) to nest under")
    ap.add_argument("--note", default="", help="free-text description of the weight edit")
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--max-pixels", type=int, default=768 * 28 * 28, help="caps image tokens to fit 4GB GPU")
    args = ap.parse_args()

    full = load_from_disk(str(DATA_DIR))
    indices = list(range(args.n))  # always the same first-N test samples, so every run is comparable
    ds = full.select(indices)

    t0 = time.perf_counter()
    proc = AutoProcessor.from_pretrained(MODEL_DIR, padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cuda").eval()  # straight to GPU: no host copy (Windows commit limit)
    n_edited = 0
    if args.delta:
        from safetensors.torch import load_file
        delta = load_file(args.delta)
        unknown = set(delta) - set(model.state_dict())
        assert not unknown, f"delta has unknown tensors: {sorted(unknown)[:5]}"
        model.load_state_dict(delta, strict=False)
        n_edited = len(delta)
    load_s = time.perf_counter() - t0

    def build(ex):
        msg = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": f"Answer the chart question with only the final answer (a number or short phrase), no explanation.\nQuestion: {ex['query']}"}]}]
        return proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    records, t1 = [], time.perf_counter()
    for i in range(0, len(ds), args.batch_size):
        exs = [ds[k] for k in range(i, min(i + args.batch_size, len(ds)))]
        images = [e["image"].convert("RGB") for e in exs]
        batch = proc(text=[build(e) for e in exs], images=images, return_tensors="pt", padding=True,
                     images_kwargs={"max_pixels": args.max_pixels}).to("cuda")
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=False)
        texts = proc.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        for ex, i_off, txt in zip(exs, range(len(exs)), texts):
            pred, target = extract_answer(txt), ex["label"][0]
            records.append({"idx": indices[i + i_off], "query": ex["query"], "target": target, "prediction": pred,
                            "correct": relaxed_correct(pred, target), "split": ex["human_or_machine"]})
    infer_s = time.perf_counter() - t1

    acc = sum(r["correct"] for r in records) / len(records)
    by_split = {}
    for r in records:
        by_split.setdefault(str(r["split"]), []).append(r["correct"])
    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": "Qwen/Qwen3.5-0.8B", "edited_tensors": n_edited, "n_samples": len(records), "batch_size": args.batch_size,
        "relaxed_accuracy": acc,
        "accuracy_by_split": {k: sum(v) / len(v) for k, v in by_split.items()},
        "model_load_seconds": round(load_s, 2), "inference_seconds": round(infer_s, 2),
        "seconds_per_sample": round(infer_s / len(records), 3),
        "estimated_full_test_minutes": round(infer_s / len(records) * 2500 / 60, 1),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"n{len(records)}_{datetime.now():%Y%m%d_%H%M%S}"
    (out / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2))
    (out / f"predictions_{tag}.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(metrics, indent=2))
    if not args.no_mlflow:
        log_mlflow(args, metrics, [out / f"metrics_{tag}.json", out / f"predictions_{tag}.json"], records)


if __name__ == "__main__":
    main()
