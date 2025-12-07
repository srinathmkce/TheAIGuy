from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from kokoro import KPipeline
import soundfile as sf
import io
import numpy as np

app = FastAPI(title="Kokoro TTS API")

# ----------------------------------------------------------------
# 1. LOAD MODEL ONCE (Global Scope)
# ----------------------------------------------------------------
# This runs only when the container starts, saving huge amounts of time per request.
print("Loading Kokoro Model... please wait.")
# lang_code='a' is American English.
pipeline = KPipeline(lang_code='a') 
print("Model Loaded! API is ready.")

# Define the data format we expect from the user
class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart" # Default voice
    speed: float = 1.0

@app.post("/v1/audio/speech")
async def generate_audio(request: TTSRequest):
    """
    Accepts text, generates audio using Kokoro, and returns a WAV file.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # Generate audio segments
        generator = pipeline(
            request.text, 
            voice=request.voice, 
            speed=request.speed, 
            split_pattern=r'\n+'
        )
        
        # Concatenate all audio segments (sentences) into one numpy array
        all_audio = []
        for _, _, audio in generator:
            all_audio.append(audio)
        
        if not all_audio:
             raise HTTPException(status_code=500, detail="No audio generated")

        final_audio = np.concatenate(all_audio)

        # Write the numpy array to an in-memory WAV file
        # We use io.BytesIO so we don't have to save to disk first
        buffer = io.BytesIO()
        sf.write(buffer, final_audio, 24000, format='WAV')
        buffer.seek(0) # Reset pointer to start of file

        # Return the audio directly as a response
        return Response(content=buffer.read(), media_type="audio/wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}