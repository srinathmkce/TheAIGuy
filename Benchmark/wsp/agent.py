"""Weight-space-programming agent for Qwen3.5-0.8B on ChartQA.

An LLM (default: OpenAI gpt-6-luna) inspects and edits the Qwen weights through a sandboxed Python tool, then calls
`run_benchmark` (the locked benchmark.py, run in a subprocess on the GPU) and iterates on the feedback. No gradient
training: edits are written directly into the tensors.

Unlike a blind "edit -> benchmark" loop, the agent works on a LIVE copy of the model on the GPU so it can do actual
weight-space programming:
  * look inside the model: next-token distributions and residual-stream activations at the answer position,
  * find behaviour directions (e.g. "start a step-by-step explanation" vs "answer directly") from contrast sets,
  * edit them out of the weights (rescale the component every layer writes along a direction),
  * measure each edit cheaply with `probe` (fixed vs broken samples vs the unedited model, with a sign test),
  * and only spend a benchmark iteration when the probe shows a clear net gain.

The agent can never touch the benchmark: it only gets the tools below, its python sandbox has no file/import
access, and the runner verifies benchmark.py + dataset + base weights are unchanged after every tool call.

Usage: uv run python agent.py --iterations 5 --n 2500 --batch-size 32
"""
import argparse, contextlib, hashlib, io, json, math, random, re, subprocess, sys, time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from dotenv import dotenv_values, load_dotenv
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText, AutoProcessor

# Langfuse reads its credentials from the environment, so load .env BEFORE importing/initialising it,
# and import the Langfuse-wrapped OpenAI client (drop-in) so every Responses call becomes a `generation`.
load_dotenv(Path(__file__).parent / ".env")
from langfuse import Langfuse, LangfuseMedia, get_client, propagate_attributes  # noqa: E402
from langfuse.openai import OpenAI  # noqa: E402

# Read-only use of the locked benchmark's scoring functions, so probe scores match the benchmark exactly.
from benchmark import _to_float, extract_answer, relaxed_correct  # noqa: E402


def _mask(data, **kwargs):
    """Drop OpenAI's opaque `encrypted_content` reasoning blobs from traces (large, unreadable noise); the readable
    reasoning *summary* is kept."""
    if isinstance(data, dict):
        return {k: ("[omitted]" if k == "encrypted_content" and v else _mask(v)) for k, v in data.items()}
    if isinstance(data, list):
        return [_mask(v) for v in data]
    return data


Langfuse(mask=_mask)  # initialise the global client with masking; get_client() below returns this instance

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "Qwen3.5-0.8B"
DATA_DIR = ROOT / "data" / "ChartQA_test"
BENCH = ROOT / "benchmark.py"
N_SAMPLES = 500
BATCH_SIZE = None  # forwarded to benchmark.py --batch-size when set (set from --batch-size in main)

# Must mirror benchmark.py exactly (checked at start-up against its source, which is never modified).
PROMPT_PREFIX = "Answer the chart question with only the final answer (a number or short phrase), no explanation.\\nQuestion: "
MAX_PIXELS, MAX_NEW_TOKENS = 768 * 28 * 28, 32
LM = "model.language_model"

TOOL_OBS = {"python": "execute-python", "probe": "probe-edit", "run_benchmark": "run-benchmark", "revert": "revert-weights",
            "finish": "finish-session"}

# ----------------------------------------------------------------------------------------------------------------------
# Error taxonomy: separates *obstacles* (format problems a weight edit can remove) from genuine capability errors.
# ----------------------------------------------------------------------------------------------------------------------
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
REASONING_START = re.compile(r"^\s*(-|\*|#|\d+\.\s|to\b|the\b|first\b|step\b|let\b|we\b|based\b|looking\b|from\b|in the\b)", re.I)
OBSTACLES = ("reasoning_truncated", "extra_text_number_ok", "extra_text_label_ok", "number_for_label")
CATEGORY_HELP = {
    "correct": "scored correct",
    "reasoning_truncated": "numeric target, model started an explanation/calculation (cut at 32 tokens or last line is not the answer)",
    "extra_text_number_ok": "numeric target, the right number is in the output but with extra text (units, '1.6 + 1.6 = 3.2')",
    "extra_text_label_ok": "text target, the right label is in the output but with extra text ('Fox 4')",
    "number_for_label": "text target (e.g. 'Which ...?'), model answered with a number",
    "wrong_number": "numeric answer, wrong value (misreading / arithmetic: capability, not format)",
    "wrong_number_with_text": "numeric target, short non-numeric or wrong-number output",
    "wrong_yes_no": "yes/no target, wrong polarity",
    "wrong_label": "text target, wrong label",
}


def categorize(pred, target):
    if relaxed_correct(pred, target):
        return "correct"
    p, t = _to_float(pred), _to_float(target)
    if t is not None:
        if p is not None:
            return "wrong_number"
        nums = NUM_RE.findall(pred)
        if any(relaxed_correct(x, target) for x in nums):
            return "extra_text_number_ok"
        if REASONING_START.search(pred) or len(pred.split()) >= 5 or any(op in pred for op in ("=", " + ", " * ", "/")):
            return "reasoning_truncated"
        return "wrong_number_with_text"
    if target.strip().lower() in ("yes", "no") and p is None:
        return "wrong_yes_no"
    if p is not None:
        return "number_for_label"
    if target.strip().lower() in pred.lower():
        return "extra_text_label_ok"
    return "wrong_label"


def sign_test(gained, lost):
    """Two-sided exact binomial test on discordant pairs (McNemar exact). Small p => the change is not noise."""
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    return round(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), 4)


def compare(new, old):
    """new/old: lists of records with idx/correct. Returns gained, lost (records from `new`)."""
    o = {r["idx"]: r for r in old}
    gained = [r for r in new if r["correct"] and r["idx"] in o and not o[r["idx"]]["correct"]]
    lost = [r for r in new if not r["correct"] and r["idx"] in o and o[r["idx"]]["correct"]]
    return gained, lost


def brief(r, old=None):
    d = {"idx": r["idx"], "query": r["query"][:160], "target": r["target"], "prediction": r["prediction"][:160]}
    if old is not None:
        d["before"] = old["prediction"][:160]
    return d


# ----------------------------------------------------------------------------------------------------------------------
# Live model on the GPU (agent side). Mirrors the benchmark's inference so probe results are representative.
# ----------------------------------------------------------------------------------------------------------------------
class TensorFile:
    """Reads single tensors from a safetensors file with plain file I/O (no mmap: friendlier to tight commit limits)."""
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


class Engine:
    def __init__(self, device, batch_size):
        self.device, self.bs = device, batch_size
        self.proc = AutoProcessor.from_pretrained(MODEL_DIR, padding_side="left")
        self.tok = self.proc.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map=device).eval()
        self.ds = load_from_disk(str(DATA_DIR))
        self.layers = self.model.get_submodule(f"{LM}.layers")
        self.params = dict(self.model.named_parameters())
        self.hidden = self.model.config.text_config.hidden_size
        self.layer_types = list(self.model.config.text_config.layer_types)

    def _batch(self, idxs):
        exs = [self.ds[int(i)] for i in idxs]
        texts = [self.proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT_PREFIX.replace("\\n", "\n") + e["query"]}]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for e in exs]
        b = self.proc(text=texts, images=[e["image"].convert("RGB") for e in exs], return_tensors="pt", padding=True,
                      images_kwargs={"max_pixels": MAX_PIXELS}).to(self.model.device)
        return exs, b

    @torch.no_grad()
    def generate(self, idxs, max_new_tokens=MAX_NEW_TOKENS):
        out = []
        for i in range(0, len(idxs), self.bs):
            _, b = self._batch(idxs[i:i + self.bs])
            g = self.model.generate(**b, max_new_tokens=max_new_tokens, do_sample=False)
            out += self.proc.batch_decode(g[:, b["input_ids"].shape[1]:], skip_special_tokens=True)
        return out

    @torch.no_grad()
    def score(self, idxs):
        """Benchmark-identical scoring of the current weights on dataset indices."""
        raws = self.generate(idxs)
        recs = []
        for i, raw in zip(idxs, raws):
            ex = self.ds[int(i)]
            pred, target = extract_answer(raw), ex["label"][0]
            recs.append({"idx": int(i), "query": ex["query"], "target": target, "prediction": pred, "raw": raw,
                         "correct": relaxed_correct(pred, target), "category": categorize(pred, target)})
        return recs

    @torch.no_grad()
    def forward_last(self, idxs, layers=None, k=0):
        """One forward pass per batch over the prompt. Returns residual-stream states at the generation position
        (output of each requested decoder layer, float32 CPU [n, len(layers), hidden]) and, if k>0, top-k next tokens."""
        layers = [] if layers is None else list(layers)
        store, hs, tops = {}, [], []

        def mk(l):
            def hook(_m, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                store[l] = h[:, -1, :].float().cpu()
            return hook
        hooks = [self.layers[l].register_forward_hook(mk(l)) for l in layers]
        try:
            for i in range(0, len(idxs), self.bs):
                _, b = self._batch(idxs[i:i + self.bs]); store.clear()
                out = self.model(**b, logits_to_keep=1, use_cache=False)
                if layers:
                    hs.append(torch.stack([store[l] for l in layers], 1))
                if k:
                    p = out.logits[:, -1].float().softmax(-1).topk(k)
                    for vals, ids in zip(p.values.tolist(), p.indices.tolist()):
                        tops.append([(self.tok.decode([t]), round(v, 4)) for t, v in zip(ids, vals)])
        finally:
            for h in hooks:
                h.remove()
        return (torch.cat(hs) if hs else None), tops


class Workspace:
    """Editable live weights + sandbox. W maps parameter name -> the model's own nn.Parameter (edits apply instantly)."""

    def __init__(self, engine, probe_idx, base_probe, base_records, max_probes):
        self.E = engine
        cache = ROOT / "cache" / "base_params.safetensors"
        if not cache.exists():  # parameter names match the module names the benchmark loads the delta into
            cache.parent.mkdir(exist_ok=True)
            save_file({k: v.detach().cpu().contiguous() for k, v in engine.params.items()}, str(cache))
        self.B = TensorFile(cache)
        assert set(self.B.keys()) == set(engine.params), "base cache does not match model parameters; delete cache/"
        self.best_state = None
        self.probe_idx, self.base_probe = probe_idx, base_probe
        self.probes_left = max_probes
        self.ns = self._namespace(base_records)

    # ---- weight state ------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def delta(self):
        d = {}
        for k, p in self.E.params.items():
            cur = p.detach().cpu()
            if not torch.equal(cur, self.B[k]):
                d[k] = cur.clone().contiguous()
        return d

    @torch.no_grad()
    def restore(self, state):
        state = state or {}
        for k in set(self.delta()) | set(state):
            p = self.E.params[k]
            p.data.copy_((state[k] if k in state else self.B[k]).to(p.device, p.dtype))

    # ---- probe -------------------------------------------------------------------------------------------------------
    def probe(self):
        if self.probes_left <= 0:
            return "ERROR: probe budget exhausted. Use run_benchmark or finish."
        self.probes_left -= 1
        t0 = time.time()
        recs = self.E.score(self.probe_idx)
        base = {r["idx"]: r for r in self.base_probe}
        gained, lost = compare(recs, self.base_probe)
        changed = [r for r in recs if r["prediction"] != base[r["idx"]]["prediction"]]
        acc, bacc = (sum(r["correct"] for r in x) / len(x) for x in (recs, self.base_probe))
        cats_now, cats_base = Counter(r["category"] for r in recs), Counter(r["category"] for r in self.base_probe)
        rng = random.Random(len(changed))
        neutral = [r for r in changed if r not in gained and r not in lost]
        return json.dumps({
            "probe_accuracy": round(acc, 4), "unedited_probe_accuracy": round(bacc, 4), "n": len(recs),
            "fixed": len(gained), "broken": len(lost), "net": len(gained) - len(lost), "sign_test_p": sign_test(len(gained), len(lost)),
            "outputs_changed": len(changed),
            "categories_now_vs_unedited": {c: [cats_now.get(c, 0), cats_base.get(c, 0)] for c in CATEGORY_HELP},
            "fixed_examples": [brief(r, base[r["idx"]]) for r in rng.sample(gained, min(6, len(gained)))],
            "broken_examples": [brief(r, base[r["idx"]]) for r in rng.sample(lost, min(8, len(lost)))],
            "other_changed_examples": [brief(r, base[r["idx"]]) for r in rng.sample(neutral, min(4, len(neutral)))],
            "probes_left": self.probes_left, "seconds": round(time.time() - t0, 1),
        }, indent=1)

    # ---- sandbox -----------------------------------------------------------------------------------------------------
    def _namespace(self, base_records):
        E, ws = self.E, self
        H = E.hidden

        class Weights:
            def keys(self): return list(E.params)
            def shape(self, k): return tuple(E.params[k].shape)
            def __contains__(self, k): return k in E.params
            def __iter__(self): return iter(E.params)
            def __len__(self): return len(E.params)
            def __getitem__(self, k): return E.params[k]
            def __setitem__(self, k, v):
                p = E.params[k]
                if tuple(v.shape) != tuple(p.shape):
                    raise ValueError(f"shape mismatch for {k}: {tuple(v.shape)} vs {tuple(p.shape)}")
                p.data.copy_(v.to(p.device, p.dtype))
            def peek(self, k): return E.params[k].detach()
            def items(self):
                for k, p in E.params.items():
                    yield k, p.detach()

        def _idx(idxs, cap):
            idxs = [int(i) for i in (idxs if isinstance(idxs, (list, tuple, range)) else [idxs])]
            if len(idxs) > cap:
                raise ValueError(f"at most {cap} indices per call")
            if any(i < 0 or i >= len(E.ds) for i in idxs):
                raise ValueError("index out of range")
            return idxs

        def run(idxs, max_new_tokens=MAX_NEW_TOKENS):
            """Raw outputs + benchmark scoring for up to 64 dataset indices with the CURRENT weights."""
            idxs = _idx(idxs, 64)
            raws = E.generate(idxs, max_new_tokens=max_new_tokens)
            res = []
            for i, raw in zip(idxs, raws):
                ex = E.ds[i]; pred = extract_answer(raw)
                res.append({"idx": i, "target": ex["label"][0], "raw": raw, "prediction": pred,
                            "correct": relaxed_correct(pred, ex["label"][0]), "category": categorize(pred, ex["label"][0])})
            return res

        def first_tokens(idxs, k=8):
            """Top-k next-token (token, prob) at the answer position for up to 128 indices."""
            return E.forward_last(_idx(idxs, 128), k=k)[1]

        def capture(idxs, layers=None):
            """Residual stream at the generation position: float32 CPU tensor [len(idxs), len(layers), hidden].
            layers defaults to all 24 (entry j = OUTPUT of decoder layer layers[j]). Up to 256 indices."""
            layers = list(range(len(E.layers))) if layers is None else list(layers)
            return E.forward_last(_idx(idxs, 256), layers=layers)[0]

        def contrast_direction(pos, neg, layer):
            """Unit vector mean(capture(pos)) - mean(capture(neg)) at `layer` (float32 CPU [hidden]), plus separation stats."""
            hp, hn = capture(pos, [layer])[:, 0], capture(neg, [layer])[:, 0]
            d = hp.mean(0) - hn.mean(0); d = d / d.norm()
            pp, pn = hp @ d, hn @ d
            thr = (pp.mean() + pn.mean()) / 2
            acc = ((pp > thr).float().sum() + (pn <= thr).float().sum()).item() / (len(pp) + len(pn))
            print(f"contrast_direction layer {layer}: proj pos {pp.mean():.2f}+-{pp.std():.2f}, neg {pn.mean():.2f}+-{pn.std():.2f}, "
                  f"train separation acc {acc:.2f}, raw diff norm {(hp.mean(0) - hn.mean(0)).norm():.2f}")
            return d

        def _writers(layers, parts):
            names = []
            for l in layers:
                if "attn" in parts:
                    names += [f"{LM}.layers.{l}.self_attn.o_proj.weight", f"{LM}.layers.{l}.linear_attn.out_proj.weight"]
                if "mlp" in parts:
                    names.append(f"{LM}.layers.{l}.mlp.down_proj.weight")
            return [n for n in names if n in E.params]

        def edit_direction(d, layers=None, factor=0.0, parts=("attn", "mlp")):
            """Rescale the component along unit direction d that the chosen layers WRITE into the residual stream:
            W_out <- W_out + (factor-1) * d d^T W_out for attention/linear-attention output projections and MLP down
            projections. factor 0 ablates d, 0.5 halves it, -0.5 inverts it partially, 1.5 amplifies it.
            Extra parts: 'merger' (visual merger fc2 weight+bias: what image tokens inject), 'embed' (token embeddings,
            which are TIED to the output layer, so this also changes output logits along d)."""
            d = torch.as_tensor(d).float().flatten().to(E.model.device)
            if d.numel() != H:
                raise ValueError(f"direction must have {H} elements")
            d = d / d.norm()
            layers = list(range(len(E.layers))) if layers is None else list(layers)
            names, tot = _writers(layers, parts), 0.0
            for n in names:
                p = E.params[n]; w = p.float()
                upd = (factor - 1.0) * torch.outer(d, d @ w)
                p.data.copy_((w + upd).to(p.dtype)); tot += upd.norm().item() ** 2
            if "merger" in parts:
                for n in ("model.visual.merger.linear_fc2.weight", "model.visual.merger.linear_fc2.bias"):
                    p = E.params[n]; w = p.float()
                    upd = (factor - 1.0) * (torch.outer(d, d @ w) if w.dim() == 2 else d * (d @ w))
                    p.data.copy_((w + upd).to(p.dtype)); names.append(n); tot += upd.norm().item() ** 2
            if "embed" in parts:
                n = f"{LM}.embed_tokens.weight"; p = E.params[n]
                for s in range(0, p.shape[0], 32768):
                    w = p[s:s + 32768].float()
                    upd = (factor - 1.0) * torch.outer(w @ d, d)
                    p.data[s:s + 32768] = (w + upd).to(p.dtype); tot += upd.norm().item() ** 2
                names.append(n)
            msg = f"edit_direction: factor {factor} on {len(names)} tensors (layers {layers[0]}..{layers[-1]}, parts {list(parts)}), update L2 {tot ** 0.5:.3f}"
            print(msg)
            return names

        def edited():
            """Names of tensors that currently differ from the base weights, with relative change ||W-B||/||B||."""
            out = {}
            for k, v in ws.delta().items():
                b = ws.B[k].float(); out[k] = round(((v.float() - b).norm() / (b.norm() + 1e-12)).item(), 5)
            return out

        S = [{k: r[k] for k in ("idx", "query", "target", "prediction", "raw", "correct", "category")} for r in self.base_probe]
        return {
            "__builtins__": SAFE_BUILTINS, "torch": torch, "np": np, "math": math, "re": re, "Counter": Counter,
            "W": Weights(), "B": self.B, "BASE": base_records, "S": S, "CATEGORIES": CATEGORY_HELP,
            "HIDDEN": H, "N_LAYERS": len(E.layers), "LAYER_TYPES": E.layer_types,
            "run": run, "first_tokens": first_tokens, "capture": capture, "contrast_direction": contrast_direction,
            "edit_direction": edit_direction, "edited": edited,
            "tok": lambda s: E.tok.encode(s, add_special_tokens=False), "detok": lambda ids: E.tok.decode(list(ids)),
            "tok_strs": lambda s: [E.tok.decode([t]) for t in E.tok.encode(s, add_special_tokens=False)],
        }

    def exec(self, code):
        if "__" in code or re.search(r"\b(import|open|exec|eval|compile|globals|locals|vars|getattr|setattr|delattr|input|breakpoint|save|load)\b", code):
            return "ERROR: forbidden construct (import/open/exec/eval/dunder/getattr/save/load...). Use the provided variables only."
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), torch.no_grad():
                exec(code, self.ns)
        except Exception as e:  # report to the agent
            buf.write(f"\nERROR: {type(e).__name__}: {e}")
        out = buf.getvalue()
        return out[:8000] + ("\n...[truncated]" if len(out) > 8000 else "") or "(no output)"


SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n) for n in
                 ("abs all any bool dict enumerate float int len list max min print range round set sorted sum tuple zip "
                  "isinstance reversed slice str map filter divmod pow repr type next iter Exception ValueError KeyError").split()}

SANDBOX_DOC = """Sandbox (python tool). Variables persist between calls.
- W: editable weights, name -> the live model's nn.Parameter (edits take effect immediately). W.keys(), W.shape(n), W.peek(n),
  W.items(). Edit in place (W[n].data[...] = ..., W[n].data.mul_(...)) or assign W[n] = tensor.
- B: untouched base tensors on CPU (B[n], B.keys(), B.shape(n)).
- BASE: unedited benchmark outputs, one dict per benchmark sample: idx, query, target, prediction, correct, category.
- S: the probe set with unedited outputs (same keys + raw = full generated text). CATEGORIES: category -> meaning.
- run(idxs, max_new_tokens=32): generate with the CURRENT weights for <=64 dataset indices -> raw text, prediction, correct, category.
- first_tokens(idxs, k=8): top-k (token, prob) for the FIRST answer token, <=128 indices.
- capture(idxs, layers=None): residual stream at the answer position, float32 [n, len(layers), HIDDEN] (output of each layer), <=256 indices.
- contrast_direction(pos_idxs, neg_idxs, layer): unit mean-difference direction at that layer (prints separation stats).
- edit_direction(d, layers=None, factor=0.0, parts=("attn","mlp")): rescale the component along d written into the residual
  stream by those layers (0 = ablate, 0.5 = halve, <0 = invert, >1 = amplify). Optional parts "merger", "embed" (tied to logits!).
- edited(): currently edited tensors with relative change. tok(s) / tok_strs(s) / detok(ids): tokenizer.
- torch, np, math, re, Counter. HIDDEN=1024, N_LAYERS=24, LAYER_TYPES (full attention at layers 3,7,11,15,19,23; others linear attention).
No imports, files, network, dunder or getattr. Output truncated to ~8000 chars, so print summaries, not tensors."""

SYSTEM = f"""You are a weight-space programmer. You improve a small vision-language model on ChartQA by analysing and directly
editing its weights. No gradient training, no external data.

Target: Qwen3.5-0.8B (bf16, 24 decoder layers, hidden 1024, tied input/output embeddings, no biases in the language model;
its RMSNorms compute x_normed * (1 + w), so w = 0 is the identity). Score: ChartQA relaxed accuracy on a fixed benchmark set
(numbers within 5%, otherwise exact case-insensitive match), greedy decoding, thinking OFF, at most 32 new tokens, and the
answer is the LAST LINE of the output. User prompt: "{PROMPT_PREFIX}<question>" with the chart image.

Where the gain is
- Most errors are genuine misreadings or arithmetic mistakes (wrong_number, wrong_label, wrong_yes_no). Weight edits remove
  obstacles; they do not add capability. Do not chase these.
- Obstacles = the output format breaks scoring although the model "knows" enough:
  * reasoning_truncated: it starts an explanation ("To find...", "1. **Blue", "- The red bar...") that hits the 32-token
    limit or ends on a non-answer line. A direct answer is often right, and right more often than a truncated one.
  * extra_text_number_ok: the right number with extra text ("53.9 hours", "1.6 + 1.6 = 3.2", "$742.07 trillion").
  * extra_text_label_ok: the right label plus a value ("Fox 4"). number_for_label: a "Which ...?" question answered with a value.
  Turning these into bare, direct answers is the realistic win, a few dozen samples. Do not break the ~70% that are correct.

Method (directional weight editing, as in refusal-direction ablation / weight-space programming)
1. Choose ONE obstacle. Build contrast sets from BASE (benchmark outputs) and S: samples showing the behaviour (e.g. category
   reasoning_truncated) vs. comparable samples answered directly (e.g. correct numeric answers, ideally to similar
   arithmetic-style questions). Look at first_tokens() to see what the model wants to emit first ("To", "1", "-", "**"...).
2. capture() activations; find the layer where the groups separate best (contrast_direction on part of the data, check the
   separation on the rest). A direction found at layer L is written by layers <= L and read by layers > L.
3. edit_direction(d, layers, factor) to remove or reduce it. Start partial (factor 0.5..0, a contiguous layer range), check a
   handful of samples with run() and first_tokens(), and make sure correct samples keep their answers.
4. probe(): free (limited count) and takes about a minute. It reports fixed vs broken samples on the probe set relative to the
   unedited model, with a sign-test p-value and category counts. Iterate steps 1-4 (layers, factor, contrast definition)
   until net fixes are clearly positive with few breaks. revert('base') or W[n] = B[n] undoes things.
5. Only then spend run_benchmark. Combine independent wins (different obstacles), but probe the combination first.
Other edits are allowed (norm gains, specific rows/columns, visual merger fc2 bias = a constant added to every image token,
token embedding rows (tied to the output logits)). Global scale sweeps (multiply a tensor or norm by 1.05) were already
tried extensively in earlier sessions and only produced noise (+-0.3 points), so do not repeat them.

Noise: one benchmark sample = 1/N. A change below ~0.5 points with similar gained and lost counts is noise. Check the
sign-test p-value in every result. Never spend benchmark runs bracketing a scale factor; do that with probe().

Tools: python (sandbox, free), probe (free, limited), run_benchmark(note) (uses 1 of a small budget, a few minutes, saves the
current edits and evaluates them with the locked benchmark), revert('best'|'base'), finish(summary).

{SANDBOX_DOC}
"""

TOOLS = [
    {"type": "function", "name": "python", "description": "Run Python in the weight-editing sandbox.\n" + SANDBOX_DOC,
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}},
    {"type": "function", "name": "probe", "description":
        "Score the CURRENT weights on the probe set (about 1 minute, does not use a benchmark iteration, limited count). "
        "Returns accuracy, fixed/broken samples vs the unedited model, sign-test p-value, category counts and examples.",
     "parameters": {"type": "object", "properties": {"note": {"type": "string", "description": "what is being tested"}}, "required": ["note"]}},
    {"type": "function", "name": "run_benchmark", "description":
        "Evaluate the current edited weights with the locked ChartQA benchmark (uses 1 iteration, takes a few minutes).",
     "parameters": {"type": "object", "properties": {"note": {"type": "string", "description": "what you changed and why"}}, "required": ["note"]}},
    {"type": "function", "name": "revert", "description": "Restore the best-scoring benchmarked edits or the original weights.",
     "parameters": {"type": "object", "properties": {"to": {"type": "string", "enum": ["best", "base"]}}, "required": ["to"]}},
    {"type": "function", "name": "finish", "description": "Stop and report.",
     "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
]


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


def check_benchmark_mirror():
    src = BENCH.read_text(encoding="utf-8")
    assert PROMPT_PREFIX in src, "agent.py PROMPT_PREFIX no longer matches benchmark.py"
    assert "768 * 28 * 28" in src and "--max-new-tokens\", type=int, default=32" in src, "benchmark defaults changed"


def run_bench(delta_path, out_dir, tag, n=N_SAMPLES, run_name=None, stage="agent_iteration", parent=None, note="", iteration=0):
    cmd = [sys.executable, str(BENCH), "--n", str(n), "--out-dir", str(out_dir), "--tag", tag]
    if BATCH_SIZE:
        cmd += ["--batch-size", str(BATCH_SIZE)]
    cmd += ["--run-name", run_name or tag, "--stage", stage, "--note", note, "--iteration", str(iteration)]
    if parent:
        cmd += ["--parent-run-id", parent]
    if delta_path:
        cmd += ["--delta", str(delta_path)]
    mfile = Path(out_dir) / f"metrics_{tag}.json"
    for attempt in range(3):  # free RAM/VRAM on a shared machine fluctuates: retry with a pause
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, encoding="utf-8", errors="replace")
        if r.returncode == 0 and mfile.exists():
            break
        time.sleep(30)
    else:
        return None, None, r.stderr[-1500:]
    return json.loads(mfile.read_text()), json.loads((Path(out_dir) / f"predictions_{tag}.json").read_text()), ""


def taxonomy(preds, k=3, seed=0):
    rng = random.Random(seed)
    by = {}
    for r in preds:
        by.setdefault(categorize(r["prediction"], r["target"]), []).append(r)
    return {c: {"count": len(by.get(c, [])), "meaning": CATEGORY_HELP[c],
                **({"examples": [brief(r) for r in rng.sample(by[c], min(k, len(by[c])))]} if c != "correct" and by.get(c) else {})}
            for c in CATEGORY_HELP}


def summarize_run(metrics, preds, p0, best_preds, base_acc):
    gained, lost = compare(preds, p0)
    base = {r["idx"]: r for r in p0}
    rng = random.Random(len(preds))
    s = {
        "relaxed_accuracy": metrics["relaxed_accuracy"], "delta_vs_baseline": round(metrics["relaxed_accuracy"] - base_acc, 4),
        "vs_baseline": {"gained": len(gained), "lost": len(lost), "sign_test_p": sign_test(len(gained), len(lost))},
        "category_counts_now_vs_baseline": {c: [sum(categorize(r["prediction"], r["target"]) == c for r in preds),
                                                sum(categorize(r["prediction"], r["target"]) == c for r in p0)] for c in CATEGORY_HELP},
        "gained_examples": [brief(r, base[r["idx"]]) for r in rng.sample(gained, min(6, len(gained)))],
        "lost_examples": [brief(r, base[r["idx"]]) for r in rng.sample(lost, min(8, len(lost)))],
    }
    if best_preds is not None:
        g, l = compare(preds, best_preds)
        s["vs_current_best"] = {"gained": len(g), "lost": len(l), "sign_test_p": sign_test(len(g), len(l))}
    return s


def choose_probe_set(n_total, n_bench, probe_n, p0, engine, seed=0):
    """Probe on held-out test samples (indices >= n_bench) when available, otherwise on the benchmark samples.
    The set is enriched with obstacle cases (half, when there are enough) so fixes are measurable, and the rest is a
    random sample so regressions on correctly answered samples show up."""
    rng = random.Random(seed)
    if n_bench + probe_n <= n_total:
        pool_idx = list(range(n_bench, min(n_total, n_bench + 4 * probe_n)))
        print(f"Probe pool: {len(pool_idx)} held-out test samples (not in the benchmark); scoring the unedited model...", flush=True)
        pool, source = engine.score(pool_idx), "held-out test samples (disjoint from the benchmark)"
    else:
        pool = [{**r, "category": categorize(r["prediction"], r["target"])} for r in p0]
        source = "benchmark samples (the benchmark covers the whole test split, so no held-out data exists)"
    obst = [r["idx"] for r in pool if r["category"] in OBSTACLES]
    rng.shuffle(obst)
    pick = obst[:probe_n // 2]
    taken = set(pick)
    rest = [r["idx"] for r in pool if r["idx"] not in taken]
    rng.shuffle(rest)
    return sorted(pick + rest[:probe_n - len(pick)]), source


def aggregate_results():
    """Combine every agent run into results/all_runs.csv (one row per run x scenario).
    Runs with an iteration_results.csv are used as-is; older runs are rebuilt from summary.json (+ metrics_iter*.json)."""
    import pandas as pd
    frames = []
    for d in sorted((ROOT / "results").glob("agent_*")):
        csv_path, summ = d / "iteration_results.csv", d / "summary.json"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        elif summ.exists():
            sj = json.loads(summ.read_text())
            base = sj["baseline_accuracy"]
            rows = []
            for h in sj["history"]:
                it = h["tune_iter"]
                mf = d / f"metrics_iter{it}.json"
                m = json.loads(mf.read_text()) if mf.exists() else {}
                rows.append({"scenario": "baseline" if it == 0 else f"iteration_{it}", "iteration": it, "accuracy": h["accuracy"],
                             "delta_vs_baseline": round(h["accuracy"] - base, 4), "edited_tensors": h.get("edited_tensors", 0),
                             "edited_parameters": h.get("edited_parameters", 0), "n_samples": m.get("n_samples", sj.get("n_samples")), "batch_size": m.get("batch_size"),
                             "inference_seconds": m.get("inference_seconds"), "peak_gpu_mem_gb": m.get("peak_gpu_mem_gb"),
                             "note": h.get("note", ""), "agent_model": sj.get("agent_model"), "timestamp": m.get("timestamp"),
                             "is_best": it == sj.get("best_iteration")})
            df = pd.DataFrame(rows)
        else:
            continue  # run produced no results (e.g. crashed before the baseline)
        df.insert(0, "run_id", d.name.removeprefix("agent_"))
        frames.append(df)
    alldf = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    alldf.to_csv(ROOT / "results" / "all_runs.csv", index=False)
    return alldf


def main():
    global BATCH_SIZE
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate-only", action="store_true", help="just rebuild results/all_runs.csv from existing runs and exit")
    ap.add_argument("--iterations", type=int, default=5, help="benchmark runs the agent may spend")
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--reasoning-summary", default="auto", choices=["auto", "concise", "detailed", "none"],
                    help="ask the model for reasoning summaries so its thinking is captured in Langfuse traces")
    ap.add_argument("--max-turns", type=int, default=250)
    ap.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens, to log estimated agent cost")
    ap.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens (incl. reasoning)")
    ap.add_argument("--batch-size", type=int, default=None, help="inference batch size for every benchmark run (default: benchmark.py's own default)")
    ap.add_argument("--n", type=int, default=N_SAMPLES, help="samples per run (use a small value for smoke tests)")
    ap.add_argument("--probe-n", type=int, default=300, help="samples in the agent's free probe set")
    ap.add_argument("--max-probes", type=int, default=40, help="probe() calls the agent may make")
    ap.add_argument("--probe-batch-size", type=int, default=None, help="batch size for the agent's live model (default: --batch-size or 16)")
    ap.add_argument("--device", default="cuda", help="device for the agent's live model")
    ap.add_argument("--offload", action="store_true", help="move the live model to CPU while the benchmark subprocess runs (small GPUs)")
    args = ap.parse_args()
    BATCH_SIZE = args.batch_size
    if args.aggregate_only:
        print(aggregate_results()[["run_id", "scenario", "accuracy", "delta_vs_baseline", "is_best"]].to_string(index=False))
        return
    check_benchmark_mirror()

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
    mlflow.log_params({"agent_model": args.model, "iterations_budget": args.iterations, "n_samples": args.n, "reasoning_effort": args.effort,
                       "probe_n": args.probe_n, "max_probes": args.max_probes})

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
    print(f"baseline accuracy = {base_acc:.4f}", flush=True)

    # Live model for the agent's analysis/probing (the benchmark still runs in its own process from a saved delta).
    engine = Engine(args.device, args.probe_batch_size or args.batch_size or 16)
    probe_idx, probe_source = choose_probe_set(len(engine.ds), args.n, args.probe_n, p0, engine)
    print(f"Probe set: {len(probe_idx)} samples from {probe_source}; scoring the unedited model...", flush=True)
    base_probe = engine.score(probe_idx)
    base_records = [{"idx": r["idx"], "query": r["query"], "target": r["target"], "prediction": r["prediction"],
                     "correct": r["correct"], "category": categorize(r["prediction"], r["target"])} for r in p0]
    ws = Workspace(engine, probe_idx, base_probe, base_records, args.max_probes)
    base_probe_acc = sum(r["correct"] for r in base_probe) / len(base_probe)
    print(f"unedited probe accuracy = {base_probe_acc:.4f}", flush=True)

    client = OpenAI(api_key=dotenv_values(ROOT / ".env")["OPENAI_API_KEY"])
    history = [{"tune_iter": 0, "note": "base", "accuracy": base_acc}]
    best = {"acc": base_acc, "iter": 0, "preds": None}
    # One row per scenario (baseline + each benchmarked iteration) -> iteration_results.csv (also attached to the Langfuse trace)
    table = [{"scenario": "baseline", "iteration": 0, "accuracy": base_acc, "delta_vs_baseline": 0.0, "edited_tensors": 0,
              "edited_parameters": 0, "n_samples": m0.get("n_samples"), "batch_size": m0.get("batch_size"), "inference_seconds": m0.get("inference_seconds"),
              "peak_gpu_mem_gb": m0.get("peak_gpu_mem_gb"), "gained_vs_baseline": 0, "lost_vs_baseline": 0, "sign_test_p": 1.0,
              "note": "unedited base weights", "agent_model": args.model, "timestamp": m0.get("timestamp")}]
    results_csv = out / "iteration_results.csv"
    def save_table():
        import pandas as pd
        df = pd.DataFrame(table)
        df["is_best"] = df["iteration"] == best["iter"]
        df.to_csv(results_csv, index=False)
        return df
    save_table()
    used, infra_fails = 0, 0
    first = {"benchmark_samples": len(p0), "baseline_relaxed_accuracy": base_acc, "baseline_error_taxonomy": taxonomy(p0),
             "probe_set": {"size": len(probe_idx), "source": probe_source, "unedited_accuracy": round(base_probe_acc, 4),
                           "categories": dict(Counter(r["category"] for r in base_probe))}}
    prompt = (f"Budget: {args.iterations} benchmark runs, {args.max_probes} probes. Baseline (unedited) result:\n"
              + json.dumps(first, indent=1)
              + "\nStart by studying the obstacle categories in BASE/S (raw outputs, first_tokens) and form a hypothesis about the "
                "behaviour you want to remove before editing anything.")
    inp, prev, finished, summary = [{"role": "user", "content": prompt}], None, False, ""

    usage = {"llm_calls": 0, "llm_input_tokens": 0, "llm_cached_input_tokens": 0, "llm_output_tokens": 0, "llm_reasoning_tokens": 0}
    root = mlflow.start_span(name="agent_session", span_type="AGENT")
    root_span = root.__enter__(); root_span.set_inputs({"model": args.model, "iterations": args.iterations, "n_samples": args.n, "baseline": base_acc})
    reasoning = {"effort": args.effort, **({"summary": args.reasoning_summary} if args.reasoning_summary != "none" else {})}
    langfuse = get_client()
    # One trace per agent run (a self-contained unit of work); the session id links it to the MLflow run of the same name.
    with propagate_attributes(trace_name="weight-space-agent-run", session_id=f"agent_{run_id}",
                              tags=["weight-space-programming", "chartqa"],
                              metadata={"agent_model": args.model, "target_model": MODEL_DIR.name, "iterations_budget": str(args.iterations),
                                        "n_samples": str(args.n), "batch_size": str(args.batch_size), "mlflow_run_id": parent}), \
         langfuse.start_as_current_observation(as_type="agent", name="run-weight-space-agent",
             input={"task": "improve ChartQA relaxed accuracy of " + MODEL_DIR.name + " by editing weights", "baseline_accuracy": base_acc,
                    "iterations_budget": args.iterations}) as lf_root:
        for turn in range(args.max_turns):
            if finished:
                break
            if turn == args.max_turns - 5:
                inp.append({"role": "user", "content": "Only 5 turns left: benchmark your best probed edit if not done yet, then finish."})
            r = client.responses.create(model=args.model, instructions=SYSTEM, input=inp, tools=TOOLS,
                                        reasoning=reasoning, previous_response_id=prev, max_output_tokens=32000,
                                        name="agent-turn", metadata={"turn": str(turn), "previous_response_id": str(prev)})
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
                inp = [{"role": "user", "content": "Continue: call a tool (python / probe / run_benchmark / revert / finish)."}]
                continue
            for c in calls:
                try:
                    args_j = json.loads(c.arguments or "{}")
                except json.JSONDecodeError:
                    args_j = {}
                L(kind="call", name=c.name, args=args_j)
                with langfuse.start_as_current_observation(as_type="tool", name=TOOL_OBS.get(c.name, "unknown-tool"),
                                                          input=args_j, metadata={"tool": c.name, "call_id": c.call_id}) as lf_tool:
                    tool_cm = mlflow.start_span(name=f"tool:{c.name}", span_type="TOOL")
                    tool_span = tool_cm.__enter__(); tool_span.set_inputs(args_j)
                    if c.name == "python":
                        res = ws.exec(args_j.get("code", ""))
                    elif c.name == "probe":
                        print(f"[probe {args.max_probes - ws.probes_left + 1}/{args.max_probes}] {args_j.get('note', '')[:150]}", flush=True)
                        res = ws.probe()
                        try:
                            pj = json.loads(res); print(f"   probe: acc {pj['probe_accuracy']:.4f} fixed {pj['fixed']} broken {pj['broken']} p={pj['sign_test_p']}", flush=True)
                        except (ValueError, KeyError):
                            pass
                    elif c.name == "revert":
                        ws.restore(ws.best_state if args_j.get("to") == "best" else None)
                        res = f"reverted to {args_j.get('to')}"
                    elif c.name == "finish":
                        finished, summary, res = True, args_j.get("summary", ""), "ok"
                    elif c.name == "run_benchmark":
                        if used >= args.iterations:
                            res = "ERROR: iteration budget exhausted. Call finish."
                        else:
                            if fingerprint() != guard:  # hashing ~2 GB, so only before each evaluation and at the end
                                raise RuntimeError("Integrity check failed: benchmark/dataset/base weights changed. Aborting.")
                            used += 1
                            delta = ws.delta()
                            dpath = out / f"iter{used}_delta.safetensors"
                            save_file(delta, str(dpath))
                            n_par = sum(t.numel() for t in delta.values())
                            print(f"[iter {used}/{args.iterations}] running benchmark: {len(delta)} tensors changed -- {args_j.get('note','')[:150]}", flush=True)
                            if args.offload:
                                engine.model.to("cpu")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            m, p, err = run_bench(dpath if delta else None, out, f"iter{used}", n=args.n,
                                                  run_name=f"iter{used}", stage="agent_iteration", parent=parent, note=args_j.get("note", ""), iteration=used)
                            if args.offload:
                                engine.model.to(args.device)
                            if m is None:  # infrastructure failure, not a result: refund the iteration
                                used -= 1; infra_fails += 1
                                lf_tool.update(level="WARNING", status_message="benchmark infrastructure failure; iteration refunded")
                                res = ("INFRASTRUCTURE ERROR (iteration refunded, not caused by your edit; just call run_benchmark again): "
                                       + err[-600:])
                                if infra_fails >= 3:
                                    raise RuntimeError(f"benchmark infrastructure failed {infra_fails} times: {err[-500:]}")
                            else:
                                s = summarize_run(m, p, p0, best["preds"], base_acc)
                                s["edited_tensors"], s["edited_parameters"] = len(delta), n_par
                                s["iterations_left"] = args.iterations - used
                                gained, lost = s["vs_baseline"]["gained"], s["vs_baseline"]["lost"]
                                history.append({"tune_iter": used, "note": args_j.get("note", ""), "accuracy": m["relaxed_accuracy"],
                                                "edited_tensors": len(delta), "edited_parameters": n_par,
                                                "gained": gained, "lost": lost, "sign_test_p": s["vs_baseline"]["sign_test_p"]})
                                table.append({"scenario": f"iteration_{used}", "iteration": used, "accuracy": m["relaxed_accuracy"],
                                              "delta_vs_baseline": round(m["relaxed_accuracy"] - base_acc, 4),
                                              "edited_tensors": len(delta), "edited_parameters": n_par, "n_samples": m.get("n_samples"), "batch_size": m.get("batch_size"),
                                              "inference_seconds": m.get("inference_seconds"), "peak_gpu_mem_gb": m.get("peak_gpu_mem_gb"),
                                              "gained_vs_baseline": gained, "lost_vs_baseline": lost, "sign_test_p": s["vs_baseline"]["sign_test_p"],
                                              "note": args_j.get("note", ""), "agent_model": args.model, "timestamp": m.get("timestamp")})
                                if m["relaxed_accuracy"] > best["acc"]:
                                    best = {"acc": m["relaxed_accuracy"], "iter": used, "preds": p}
                                    ws.best_state = delta
                                    s["new_best"] = True
                                save_table()
                                lf_tool.update(metadata={"iteration": str(used), "edited_tensors": str(len(delta)), "edited_parameters": str(n_par)})
                                lf_tool.score(name="relaxed_accuracy", value=m["relaxed_accuracy"], comment=args_j.get("note", "")[:500])
                                print(f"[iter {used}] accuracy {m['relaxed_accuracy']:.4f} (base {base_acc:.4f}, best {best['acc']:.4f}) "
                                      f"gained {gained} lost {lost} p={s['vs_baseline']['sign_test_p']}", flush=True)
                                res = json.dumps(s, indent=1)
                    else:
                        res = f"unknown tool {c.name}"
                    L(kind="result", name=c.name, result=res[:4000])
                    tool_span.set_outputs(res[:4000])
                    lf_tool.update(output=res[:4000])
                    tool_cm.__exit__(None, None, None)
                    inp.append({"type": "function_call_output", "call_id": c.call_id, "output": res})
        if fingerprint() != guard:
            raise RuntimeError("Integrity check failed: benchmark/dataset/base weights changed. Aborting.")
        df = save_table()
        print("\n" + df[["scenario", "accuracy", "delta_vs_baseline", "gained_vs_baseline", "lost_vs_baseline", "sign_test_p", "is_best"]].to_string(index=False), flush=True)
        lf_root.update(metadata={"iteration_results_csv": LangfuseMedia(content_bytes=results_csv.read_bytes(),
                                                                        content_type="text/csv")})
        lf_root.update(output={"best_accuracy": best["acc"], "best_iteration": best["iter"], "baseline_accuracy": base_acc,
                               "iterations_used": used, "history": history, "agent_summary": summary})
        lf_root.score_trace(name="best_relaxed_accuracy", value=best["acc"])
        lf_root.score_trace(name="improvement_vs_baseline", value=round(best["acc"] - base_acc, 4))

    root.__exit__(None, None, None)
    if ws.best_state:
        save_file(ws.best_state, str(out / "best_delta.safetensors"))
    best_sig = {}
    if best["preds"] is not None:
        g, l = compare(best["preds"], p0)
        best_sig = {"gained": len(g), "lost": len(l), "sign_test_p": sign_test(len(g), len(l))}
    probes_used = args.max_probes - ws.probes_left
    del ws, engine
    final = {
        "run_id": run_id, "agent_model": args.model, "n_samples": args.n, "iterations_used": used,
        "probe_set": {"size": len(probe_idx), "source": probe_source, "probes_used": probes_used,
                      "unedited_accuracy": round(base_probe_acc, 4)},
        "baseline_accuracy": base_acc, "best_accuracy": best["acc"], "best_iteration": best["iter"],
        "improvement": round(best["acc"] - base_acc, 4), "best_vs_baseline_significance": best_sig,
        "history": history, "agent_summary": summary,
    }
    (out / "summary.json").write_text(json.dumps(final, indent=2))
    mlflow.log_metrics({"baseline_accuracy": base_acc, "best_accuracy": best["acc"], "improvement": final["improvement"],
                        "iterations_used": used, **({"best_sign_test_p": best_sig["sign_test_p"]} if best_sig else {})})
    mlflow.set_tags({"best_iteration": str(best["iter"])})
    cost = {}
    if args.price_in is not None and args.price_out is not None:
        cost["agent_est_cost_usd"] = (usage["llm_input_tokens"] - usage["llm_cached_input_tokens"]) * args.price_in / 1e6 \
            + usage["llm_output_tokens"] * args.price_out / 1e6
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
    mlflow.log_artifact(str(results_csv))
    aggregate_results()  # refresh results/all_runs.csv now that this run's CSV exists
    mlflow.end_run()
    langfuse.flush()  # short-lived script: make sure all spans are sent before exit
    print(json.dumps({k: v for k, v in final.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
