# Streaming ASR Chatbot

Click the mic, talk, and see the transcription stream live — powered by
[nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
via HuggingFace Transformers.

## Architecture

- `backend/` — ASR business logic (model loading, `StreamingSession`: one
  continuous `model.generate()` call per recording, fed by a live audio
  queue). No FastAPI imports.
- `api/` — FastAPI app: WebSocket route (`/ws/transcribe`), CORS, health
  check. Thin adapter over `backend/`.
- `frontend/` — React + TypeScript + Vite UI: mic capture via AudioWorklet,
  WebSocket client, live transcript view.
- `scripts/smoke_test_stream.py` — exercises `StreamingSession` directly
  against a test audio file, no browser needed.

## Setup

Backend (from `streaming/`):
```
uv sync
uv run python scripts/smoke_test_stream.py   # optional smoke test
uv run python main.py                         # serves on :8000
```

Frontend (from `streaming/frontend/`):
```
npm install
npm run dev                                   # serves on :5173
```

## Scope / known limitations

- Single concurrent session: the model is a shared GPU singleton, so a
  second WebSocket connection while one session is active is rejected with
  a "busy" error rather than queued.
- ASR-only: this app streams a live transcript, it does not send the
  transcript to an LLM for a reply.
