# YouTube Chatbot

Ask questions about any YouTube video based on its transcript. Paste a URL, see the thumbnail, and chat with the video content — powered by GPT-4o-mini.

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.11+ | [python.org](https://www.python.org/downloads/) |
| uv | latest | `pip install uv` or [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/) |
| Node.js | 18+ | [nodejs.org](https://nodejs.org/) |
| OpenAI API key | — | [platform.openai.com](https://platform.openai.com/api-keys) |

## Setup

### 1. Clone the repo

```bash
git clone <repo-url>
cd youtube-chatbot
```

### 2. Configure your OpenAI API key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-...
```

### 3. Install Python dependencies

```bash
uv sync
```

### 4. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

## Running the App

Open two terminals from the project root.

**Terminal 1 — Backend (FastAPI):**

```bash
uv run uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — Frontend (React + Vite):**

```bash
cd frontend
npm run dev
```

Open your browser at **http://localhost:5173**.

## How to Use

1. Paste a YouTube video URL into the input field — the thumbnail appears instantly.
2. Type a question about the video in the chat bar and press **Send**.
3. The backend fetches the video's captions and asks GPT-4o-mini to answer based on the transcript.

## Supported YouTube URL Formats

```
https://www.youtube.com/watch?v=VIDEO_ID
https://youtube.com/watch?v=VIDEO_ID
https://youtu.be/VIDEO_ID
https://www.youtube.com/embed/VIDEO_ID
```

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Invalid YouTube URL` | URL format not recognised | Use one of the supported formats above |
| `No captions/transcripts available` | Video has no captions | Try a different video that has auto-generated or manual captions |
| `OpenAI error` | Bad API key or quota exceeded | Check `OPENAI_API_KEY` in `.env` and your OpenAI account |
| Backend not reachable | Backend not running | Start `uvicorn` in Terminal 1 first |

## Project Structure

```
youtube-chatbot/
├── backend/
│   ├── main.py       # FastAPI app, /chat endpoint
│   └── utils.py      # extract_video_id, get_transcript, ask_gpt
├── frontend/
│   ├── src/
│   │   ├── App.jsx   # UI: URL input, thumbnail, chat
│   │   └── App.css   # Styling
│   ├── index.html
│   └── package.json
├── .env              # OPENAI_API_KEY (not committed)
└── pyproject.toml    # Python dependencies
```
