"""
ASR Metrics — reusable benchmarking script for automatic speech recognition.

Usage (CLI):
    python asr_metrics.py --reference ref.txt --hypothesis hyp.txt
    python asr_metrics.py --reference ref.txt --hypothesis hyp1.txt hyp2.txt --labels ModelA ModelB
    python asr_metrics.py --reference ref.txt --hypothesis hyp.txt --json

Usage (module):
    from asr_metrics import compute_metrics, load_transcript
    metrics = compute_metrics(reference_text, hypothesis_text)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from collections import Counter
from typing import Optional


# ── Text normalisation ────────────────────────────────────────────────────────

_SPEAKER_TS = re.compile(r"^.+\(\d{2}:\d{2}\)\s*$")


def _clean_transcript(text: str, strip_speaker_lines: bool = True) -> str:
    """Lower-level cleaning: strip speaker/timestamp headers, collapse whitespace."""
    lines = text.split("\n")
    if strip_speaker_lines:
        lines = [l for l in lines if not _SPEAKER_TS.match(l.strip())]
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _tokenise(text: str) -> list[tuple[str, str]]:
    """Return (original, normalised) pairs for each whitespace token."""
    result = []
    for orig in text.strip().split():
        norm = re.sub(r"[^a-z0-9']", "", orig.lower())
        if norm:
            result.append((orig, norm))
    return result


# ── Edit-distance (word level) ────────────────────────────────────────────────

_OP_M, _OP_S, _OP_D, _OP_I = 1, 2, 3, 4


def _word_ops(
    ref: list[tuple[str, str]], hyp: list[tuple[str, str]]
) -> list[tuple[str, Optional[tuple], Optional[tuple]]]:
    """Return aligned (op, ref_tok, hyp_tok) triples via dynamic programming."""
    n, m = len(ref), len(hyp)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    op   = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        cost[i][0] = i
        op[i][0] = _OP_D
    for j in range(m + 1):
        cost[0][j] = j
        op[0][j] = _OP_I

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1][1] == hyp[j - 1][1]:
                cost[i][j] = cost[i - 1][j - 1]
                op[i][j] = _OP_M
            else:
                s   = cost[i - 1][j - 1] + 1
                d   = cost[i - 1][j]     + 1
                ins = cost[i][j - 1]     + 1
                if s <= d and s <= ins:
                    cost[i][j] = s; op[i][j] = _OP_S
                elif d <= ins:
                    cost[i][j] = d; op[i][j] = _OP_D
                else:
                    cost[i][j] = ins; op[i][j] = _OP_I

    ops: list[tuple[str, Optional[tuple], Optional[tuple]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i == 0:
            ops.append(("I", None, hyp[j - 1])); j -= 1
        elif j == 0:
            ops.append(("D", ref[i - 1], None)); i -= 1
        else:
            o = op[i][j]
            if o == _OP_M:
                ops.append(("M", ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
            elif o == _OP_S:
                ops.append(("S", ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
            elif o == _OP_D:
                ops.append(("D", ref[i - 1], None)); i -= 1
            else:
                ops.append(("I", None, hyp[j - 1])); j -= 1
    ops.reverse()
    return ops


# ── Edit-distance (character level) ──────────────────────────────────────────

def _char_edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b):
            curr.append(prev[j] if ca == cb else 1 + min(prev[j], prev[j + 1], curr[-1]))
        prev = curr
    return prev[-1]


# ── Public API ────────────────────────────────────────────────────────────────

@dataclass
class ASRMetrics:
    # Counts
    ref_words: int
    hyp_words: int
    matches: int
    substitutions: int
    deletions: int
    insertions: int
    # Rates
    wer: float          # Word Error Rate
    cer: float          # Character Error Rate
    word_accuracy: float
    mer: float          # Match Error Rate
    wil: float          # Word Information Lost
    sub_rate: float
    del_rate: float
    ins_rate: float
    # Substitution pairs sorted by frequency
    top_substitutions: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def load_transcript(path: str | Path, strip_speaker_lines: bool = True) -> str:
    """Read a transcript file and return cleaned text."""
    text = Path(path).read_text(encoding="utf-8")
    return _clean_transcript(text, strip_speaker_lines=strip_speaker_lines)


def compute_metrics(
    reference: str,
    hypothesis: str,
    strip_speaker_lines: bool = True,
    top_n_subs: int = 15,
) -> ASRMetrics:
    """
    Compute ASR quality metrics between a reference and hypothesis transcript.

    Parameters
    ----------
    reference  : ground-truth text (raw or pre-cleaned)
    hypothesis : model output text
    strip_speaker_lines : remove `Name (MM:SS)` header lines from reference
    top_n_subs : how many top substitution pairs to return
    """
    ref_clean = _clean_transcript(reference, strip_speaker_lines=strip_speaker_lines)
    hyp_clean = _clean_transcript(hypothesis, strip_speaker_lines=False)

    ref_toks = _tokenise(ref_clean)
    hyp_toks = _tokenise(hyp_clean)

    ops = _word_ops(ref_toks, hyp_toks)

    S = sum(1 for o in ops if o[0] == "S")
    D = sum(1 for o in ops if o[0] == "D")
    I = sum(1 for o in ops if o[0] == "I")
    M = sum(1 for o in ops if o[0] == "M")
    N = len(ref_toks)
    H = len(hyp_toks)

    ref_norm = "".join(t[1] for t in ref_toks)
    hyp_norm = "".join(t[1] for t in hyp_toks)
    cer = _char_edit(ref_norm, hyp_norm) / max(len(ref_norm), 1)

    sub_counts: Counter = Counter()
    for o in ops:
        if o[0] == "S":
            sub_counts[(o[1][1], o[2][1])] += 1  # type: ignore[index]

    top_subs = [
        {"ref": r, "hyp": h, "count": cnt}
        for (r, h), cnt in sub_counts.most_common(top_n_subs)
    ]

    return ASRMetrics(
        ref_words=N,
        hyp_words=H,
        matches=M,
        substitutions=S,
        deletions=D,
        insertions=I,
        wer=(S + D + I) / max(N, 1),
        cer=cer,
        word_accuracy=M / max(N, 1),
        mer=(S + D + I) / max(S + D + I + M, 1),
        wil=1 - (M / max(N, 1)) * (M / max(H, 1)),
        sub_rate=S / max(N, 1),
        del_rate=D / max(N, 1),
        ins_rate=I / max(N, 1),
        top_substitutions=top_subs,
    )


# ── CLI formatting ────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    return "#" * filled + "." * (width - filled)


def _rating(rate: float) -> str:
    if rate <= 0.15:
        return "GOOD"
    if rate <= 0.35:
        return "WARN"
    return "BAD "


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _out(text: str = "") -> None:
    """Print with UTF-8-safe output (handles Windows cp1252 terminals)."""
    sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))


def print_report(metrics: ASRMetrics, label: str = "Model") -> None:
    w = 60
    sep = "-" * w

    _out(f"\n{'=' * w}")
    _out(f"  ASR METRICS REPORT -- {label}")
    _out(f"{'=' * w}")

    _out(f"\n  PRIMARY METRICS")
    _out(f"  {sep}")
    _out(f"  {'WER  (Word Error Rate)':<30} {_pct(metrics.wer):>7}  [{_rating(metrics.wer)}]  {_bar(metrics.wer)}")
    _out(f"  {'CER  (Char Error Rate)':<30} {_pct(metrics.cer):>7}  [{_rating(metrics.cer)}]  {_bar(metrics.cer)}")
    _out(f"  {'Word Accuracy':<30} {_pct(metrics.word_accuracy):>7}  [{_rating(1 - metrics.word_accuracy)}]  {_bar(metrics.word_accuracy)}")

    _out(f"\n  ADDITIONAL METRICS")
    _out(f"  {sep}")
    _out(f"  {'MER  (Match Error Rate)':<30} {_pct(metrics.mer):>7}  [{_rating(metrics.mer)}]")
    _out(f"  {'WIL  (Word Info Lost)':<30} {_pct(metrics.wil):>7}  [{_rating(metrics.wil)}]")

    _out(f"\n  EDIT OPERATIONS")
    _out(f"  {sep}")
    _out(f"  {'Reference words':<30} {metrics.ref_words:>7}")
    _out(f"  {'Hypothesis words':<30} {metrics.hyp_words:>7}  ({metrics.hyp_words - metrics.ref_words:+d} vs ref)")
    _out(f"  {'Matches':<30} {metrics.matches:>7}  ({_pct(metrics.matches / max(metrics.ref_words, 1))})")
    _out(f"  {'Substitutions':<30} {metrics.substitutions:>7}  (sub rate: {_pct(metrics.sub_rate)})")
    _out(f"  {'Deletions':<30} {metrics.deletions:>7}  (del rate: {_pct(metrics.del_rate)})")
    _out(f"  {'Insertions':<30} {metrics.insertions:>7}  (ins rate: {_pct(metrics.ins_rate)})")

    if metrics.top_substitutions:
        _out(f"\n  TOP SUBSTITUTIONS")
        _out(f"  {sep}")
        _out(f"  {'#':<4} {'Reference':^20} {'Hypothesis':^20} {'Count':>5}")
        _out(f"  {'-'*4} {'-'*20} {'-'*20} {'-'*5}")
        for i, sub in enumerate(metrics.top_substitutions[:10], 1):
            _out(f"  {i:<4} {sub['ref']:^20} {sub['hyp']:^20} {sub['count']:>5}")

    _out(f"\n{'=' * w}\n")


def print_comparison_table(results: list[tuple[str, ASRMetrics]]) -> None:
    """Print a side-by-side comparison of multiple models."""
    w = 70
    _out(f"\n{'=' * w}")
    _out(f"  BENCHMARK COMPARISON")
    _out(f"{'=' * w}")

    col_w = 12
    header = f"  {'Metric':<22}" + "".join(f"{label:>{col_w}}" for label, _ in results)
    _out(header)
    _out(f"  {'-' * 22}" + "-" * (col_w * len(results)))

    rows = [
        ("WER",           lambda m: _pct(m.wer)),
        ("CER",           lambda m: _pct(m.cer)),
        ("Word Accuracy", lambda m: _pct(m.word_accuracy)),
        ("MER",           lambda m: _pct(m.mer)),
        ("WIL",           lambda m: _pct(m.wil)),
        ("Ref Words",     lambda m: str(m.ref_words)),
        ("Hyp Words",     lambda m: str(m.hyp_words)),
        ("Matches",       lambda m: str(m.matches)),
        ("Substitutions", lambda m: str(m.substitutions)),
        ("Deletions",     lambda m: str(m.deletions)),
        ("Insertions",    lambda m: str(m.insertions)),
    ]

    for name, fn in rows:
        row = f"  {name:<22}" + "".join(f"{fn(m):>{col_w}}" for _, m in results)
        _out(row)

    print(f"{'═' * w}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute WER/CER and related ASR metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--reference", "-r", required=True,
        help="Path to ground-truth transcript file",
    )
    parser.add_argument(
        "--hypothesis", "-y", nargs="+", required=True,
        help="Path(s) to model hypothesis transcript file(s)",
    )
    parser.add_argument(
        "--labels", "-l", nargs="+",
        help="Display labels for each hypothesis (default: filename stems)",
    )
    parser.add_argument(
        "--no-strip-speaker", action="store_true",
        help="Do not strip 'Speaker (MM:SS)' lines from reference",
    )
    parser.add_argument(
        "--top-subs", type=int, default=15,
        help="Number of top substitution pairs to show (default: 15)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON instead of human-readable text",
    )
    args = parser.parse_args()

    ref_text = load_transcript(args.reference, strip_speaker_lines=not args.no_strip_speaker)

    labels = args.labels or [Path(h).stem for h in args.hypothesis]
    if len(labels) < len(args.hypothesis):
        labels += [Path(h).stem for h in args.hypothesis[len(labels):]]

    results: list[tuple[str, ASRMetrics]] = []
    for hyp_path, label in zip(args.hypothesis, labels):
        hyp_text = Path(hyp_path).read_text(encoding="utf-8")
        m = compute_metrics(
            ref_text, hyp_text,
            strip_speaker_lines=False,
            top_n_subs=args.top_subs,
        )
        results.append((label, m))

    if args.json:
        output = {label: m.to_dict() for label, m in results}
        print(json.dumps(output, indent=2))
        return

    if len(results) == 1:
        label, m = results[0]
        print_report(m, label=label)
    else:
        for label, m in results:
            print_report(m, label=label)
        print_comparison_table(results)


if __name__ == "__main__":
    main()
