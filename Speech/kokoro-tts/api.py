import io
import numpy as np
import soundfile as sf
import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from kokoro import KPipeline

# --- 1. SETUP MODEL (Global Scope) ---
print("Loading Kokoro Model... please wait.")
# lang_code='a' is American English.
pipeline = KPipeline(lang_code="a")
print("Model Loaded! System ready.")


# --- 2. DEFINE CORE GENERATION FUNCTION ---
def generate_audio_numpy(text, voice, speed):
    """
    Helper function used by both API and UI.
    Returns: (sample_rate, audio_numpy_array)
    """
    generator = pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+")
    all_audio = []
    for _, _, audio in generator:
        all_audio.append(audio)

    if not all_audio:
        return None

    return 24000, np.concatenate(all_audio)


# --- 3. FASTAPI APP ---
app = FastAPI(title="Kokoro TTS API")


class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    speed: float = 1.0


@app.post("/v1/audio/speech")
async def api_generate_audio(request: TTSRequest):
    result = generate_audio_numpy(request.text, request.voice, request.speed)
    if not result:
        raise HTTPException(status_code=500, detail="No audio generated")

    sample_rate, audio_data = result

    # Convert to WAV in-memory
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")
    buffer.seek(0)
    return Response(content=buffer.read(), media_type="audio/wav")


@app.get("/health")
def health_check():
    return {"status": "ok"}


# --- 4. GRADIO UI ---
# We define the UI layout here
with gr.Blocks(title="Kokoro TTS (Local Docker)") as demo:
    gr.Markdown("# 🦜 Kokoro TTS - Local CPU Docker")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Input Text", lines=3, placeholder="Type something here..."
            )
            with gr.Row():
                voice_dropdown = gr.Dropdown(
                    choices=[   
                        "af_heart",
                        "af_alloy",
                        "af_aoede",
                        "af_bella",
                        "af_jessica",
                        "af_kore",
                        "af_nicole",
                        "af_nova",
                        "af_river",
                        "af_sarah",
                        "af_sky",
                        "am_adam",
                        "am_echo",
                        "am_eric",
                        "am_fenrir",
                        "am_liam",
                        "am_michael",
                        "am_onyx",
                        "am_puck",
                        "am_santa",
                        "bf_alice",
                        "bf_emma",
                        "bf_isabella",
                        "bf_lily",
                        "bm_daniel",
                        "bm_fable",
                        "bm_george",
                        "bm_lewis",
                    ],
                    value="af_heart",
                    label="Voice",
                )
                speed_slider = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Speed")

            submit_btn = gr.Button("Generate Audio", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio", type="numpy")

    # Connect the UI elements to the function
    # Note: Gradio expects the function to return (sample_rate, numpy_array) for audio
    submit_btn.click(
        fn=generate_audio_numpy,
        inputs=[text_input, voice_dropdown, speed_slider],
        outputs=[audio_output],
    )

# --- 5. MOUNT GRADIO ON FASTAPI ---
# This makes the UI accessible at the root URL "/"
app = gr.mount_gradio_app(app, demo, path="/")
