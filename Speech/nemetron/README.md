# Nemotron Speech — ASR Benchmark

Streaming speech-to-text with **NVIDIA Nemotron** (`nvidia/nemotron-speech-streaming-en-0.6b`) and a built-in benchmarking toolkit that measures WER, CER, and more against a ground-truth transcript.

---

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- A GPU is recommended (model uses `device_map="auto"`)

---

## Setup

```bash
uv sync
```

---

## Usage

### 1. Live transcription UI (real-time mic input)

A Streamlit frontend + FastAPI backend that records from your microphone, sends audio chunks to the model, and streams transcription tokens back to the browser as you speak.

**Start the backend** (loads the model, exposes a WebSocket at `ws://localhost:8000/ws/transcribe`):

```bash
uv run python backend.py
```

**Start the frontend** (in a second terminal):

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501`, click **Start Recording**, speak, and watch the transcription update every 3 seconds. Click **Stop Recording** when done. A **Download Transcription** button appears at the bottom once recording stops.

> **First run:** the model (~600 MB) is downloaded from Hugging Face automatically. A GPU is used if CUDA is available (`device_map="auto"`).

---

### 2. Run streaming transcription (file-based)

Transcribes an audio file in real time and writes the output to `transcription_output.txt`.

```bash
uv run python nemo_benchmark.py
```

> Edit the audio file path inside `nemo_benchmark.py` to point to your own file.

The script prints tokens to the console as they are generated and saves the full transcript when done.

---

### 2. Evaluate transcription quality

Compare the model output against a ground-truth reference using `asr_metrics.py`.

**Single model:**
```bash
uv run python asr_metrics.py \
  --reference transcript.txt \
  --hypothesis transcription_output.txt
```

**Multiple models side-by-side:**
```bash
uv run python asr_metrics.py \
  --reference transcript.txt \
  --hypothesis modelA_output.txt modelB_output.txt \
  --labels "Nemotron" "Whisper"
```

**JSON output (for programmatic use):**
```bash
uv run python asr_metrics.py \
  --reference transcript.txt \
  --hypothesis transcription_output.txt \
  --json
```

**All options:**

| Flag | Description |
|---|---|
| `--reference` / `-r` | Path to the ground-truth transcript |
| `--hypothesis` / `-y` | Path(s) to model output file(s) |
| `--labels` / `-l` | Display names for each model (defaults to filename) |
| `--top-subs` | Number of top substitution pairs to show (default: 15) |
| `--no-strip-speaker` | Keep `Speaker (MM:SS)` header lines in the reference |
| `--json` | Print results as JSON instead of a formatted report |

---

## Metrics explained

| Metric | What it measures | Lower is better |
|---|---|---|
| **WER** | Word Error Rate — `(S + D + I) / N` | Yes |
| **CER** | Character Error Rate — edit distance at character level | Yes |
| **Word Accuracy** | `Matches / Reference words` | No (higher = better) |
| **MER** | Match Error Rate — errors as a share of all aligned words | Yes |
| **WIL** | Word Information Lost — penalises both missed and extra words | Yes |

Edit operations counted:
- **S** — Substitution (wrong word)
- **D** — Deletion (word missing from hypothesis)
- **I** — Insertion (extra word in hypothesis)
- **M** — Match (correct word)

**Rating thresholds used in the report:**

| Score | Rating |
|---|---|
| ≤ 15% | `GOOD` |
| 16–35% | `WARN` |
| > 35% | `BAD` |

---

## Sample report output

```
============================================================
  ASR METRICS REPORT -- Nemotron
============================================================

  PRIMARY METRICS
  ------------------------------------------------------------
  WER  (Word Error Rate)          12.3%  [GOOD]  ###.................
  CER  (Char Error Rate)           8.1%  [GOOD]  ##..................
  Word Accuracy                   87.7%  [GOOD]  #################...

  ADDITIONAL METRICS
  ------------------------------------------------------------
  MER  (Match Error Rate)         11.5%  [GOOD]
  WIL  (Word Info Lost)           13.2%  [GOOD]

  EDIT OPERATIONS
  ------------------------------------------------------------
  Reference words                   412
  Hypothesis words                  398  (-14 vs ref)
  Matches                           361  (87.6%)
  Substitutions                      22  (sub rate: 5.3%)
  Deletions                          29  (del rate: 7.0%)
  Insertions                          9  (ins rate: 2.2%)

  TOP SUBSTITUTIONS
  ------------------------------------------------------------
  #    Reference              Hypothesis             Count
  ---- -------------------- -------------------- -----
  1         the                    a                  3
  ...
```

---

## Use as a Python module

```python
from asr_metrics import compute_metrics, load_transcript

ref  = load_transcript("transcript.txt")
hyp  = open("transcription_output.txt").read()

m = compute_metrics(ref, hyp)
print(f"WER: {m.wer:.1%}  |  CER: {m.cer:.1%}  |  Accuracy: {m.word_accuracy:.1%}")
```

---

## Project structure

```
nemetron/
├── app.py                 # Streamlit frontend (live mic UI)
├── backend.py             # FastAPI + WebSocket transcription server
├── nemo_benchmark.py      # Streaming transcription script (file-based)
├── asr_metrics.py         # Metrics computation and CLI
├── main.py                # Entry point placeholder
├── transcript.txt         # Ground-truth reference transcript
├── transcription_output.txt  # Model output (generated)
├── transcription_analysis.html  # Visual HTML report (generated)
└── pyproject.toml         # Dependencies (managed with uv)
```

---

## Model

**[nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)**
— A 0.6B parameter RNN-T model optimised for low-latency streaming English ASR.
