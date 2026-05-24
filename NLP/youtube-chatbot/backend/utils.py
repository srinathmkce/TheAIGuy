import os
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()


def extract_video_id(url: str) -> str | None:
    try:
        parsed = urlparse(url.strip())
        if parsed.hostname == "youtu.be":
            return parsed.path.lstrip("/").split("?")[0] or None
        if parsed.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
            if parsed.path == "/watch":
                return parse_qs(parsed.query).get("v", [None])[0]
            if parsed.path.startswith(("/embed/", "/v/")):
                return parsed.path.split("/")[2] or None
    except Exception:
        pass
    return None


def get_transcript(video_id: str) -> str:
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)
    transcript = next(iter(transcript_list))
    fetched = transcript.fetch()
    return " ".join(snippet.text for snippet in fetched)


def ask_gpt(transcript: str, question: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions about YouTube videos. "
                    "Answer based only on the transcript provided. "
                    "If the answer is not in the transcript, say so clearly."
                ),
            },
            {
                "role": "user",
                "content": f'Transcript:\n"""\n{transcript}\n"""\n\nQuestion: {question}',
            },
        ],
    )
    return response.choices[0].message.content
