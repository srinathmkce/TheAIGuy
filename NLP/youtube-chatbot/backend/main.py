from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

from backend.utils import extract_video_id, get_transcript, ask_gpt

app = FastAPI(title="YouTube Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    video_url: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    video_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    video_id = extract_video_id(req.video_url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    try:
        transcript = get_transcript(video_id)
    except (TranscriptsDisabled, NoTranscriptFound):
        raise HTTPException(
            status_code=422,
            detail="This video does not have captions/transcripts available.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch transcript: {e}")

    try:
        answer = ask_gpt(transcript, req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")

    return ChatResponse(answer=answer, video_id=video_id)
