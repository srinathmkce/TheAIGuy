import asyncio
import io
import numpy as np
import soundfile as sf
import torch
from threading import Thread
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForRNNT, AutoProcessor, TextIteratorStreamer

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"

processor: AutoProcessor = None
model: AutoModelForRNNT = None


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def load_model():
    global processor, model
    print(f"[backend] loading model on {DEVICE} (dtype={DTYPE})")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.set_num_lookahead_tokens(6)
    model = AutoModelForRNNT.from_pretrained(MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE)


@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_model)


def decode_audio(raw_bytes: bytes, target_sr: int) -> np.ndarray:
    """Decode raw PCM float32 bytes or audio file bytes into a float32 numpy array."""
    buf = io.BytesIO(raw_bytes)
    try:
        audio, sr = sf.read(buf, dtype="float32")
    except Exception:
        audio = np.frombuffer(raw_bytes, dtype=np.float32)
        sr = target_sr
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def make_feature_generator(audio: np.ndarray, first_chunk_inputs):
    def generator():
        yield first_chunk_inputs.input_features[:, : processor.num_mel_frames_first_audio_chunk, :]
        mel_frame_idx = processor.num_mel_frames_first_audio_chunk
        hop_length = processor.feature_extractor.hop_length
        n_fft = processor.feature_extractor.n_fft
        start_idx = mel_frame_idx * hop_length - n_fft // 2
        while (end_idx := start_idx + processor.num_samples_per_audio_chunk) < audio.shape[0]:
            inputs = processor(
                audio[start_idx:end_idx],
                sampling_rate=processor.feature_extractor.sampling_rate,
                is_streaming=True,
                is_first_audio_chunk=False,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device, dtype=model.dtype)
            yield inputs.input_features
            mel_frame_idx += processor.num_mel_frames_per_audio_chunk
            start_idx = mel_frame_idx * hop_length - n_fft // 2
    return generator


@app.websocket("/ws/transcribe")
async def transcribe_ws(websocket: WebSocket):
    await websocket.accept()
    sampling_rate = processor.feature_extractor.sampling_rate
    loop = asyncio.get_running_loop()

    try:
        while True:
            raw = await websocket.receive_bytes()
            if not raw:
                continue

            print(f"[backend] received {len(raw)} bytes")
            token_queue: asyncio.Queue = asyncio.Queue()

            def run_transcription(audio_bytes):
                try:
                    audio = decode_audio(audio_bytes, sampling_rate)
                    print(f"[backend] decoded → {audio.shape[0]} samples ({audio.shape[0]/sampling_rate:.2f}s)")

                    if audio.shape[0] < processor.num_samples_first_audio_chunk:
                        pad = np.zeros(
                            processor.num_samples_first_audio_chunk - audio.shape[0],
                            dtype=np.float32,
                        )
                        audio = np.concatenate([audio, pad])

                    first_chunk_inputs = processor(
                        audio[: processor.num_samples_first_audio_chunk],
                        sampling_rate=sampling_rate,
                        is_streaming=True,
                        is_first_audio_chunk=True,
                        return_tensors="pt",
                    )
                    first_chunk_inputs = first_chunk_inputs.to(model.device, dtype=model.dtype)

                    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)
                    gen_kwargs = {
                        **first_chunk_inputs,
                        "input_features": make_feature_generator(audio, first_chunk_inputs)(),
                        "streamer": streamer,
                    }
                    t = Thread(target=model.generate, kwargs=gen_kwargs)
                    t.start()
                    for chunk in streamer:
                        loop.call_soon_threadsafe(token_queue.put_nowait, chunk)
                    t.join()
                    print("[backend] transcription complete")
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    loop.call_soon_threadsafe(token_queue.put_nowait, f"[Error: {exc}]")
                finally:
                    # Always send sentinel so the async reader unblocks
                    loop.call_soon_threadsafe(token_queue.put_nowait, None)

            executor_future = loop.run_in_executor(None, run_transcription, raw)

            # Stream tokens to the client as they are generated
            while True:
                token = await asyncio.wait_for(token_queue.get(), timeout=120.0)
                if token is None:
                    break
                await websocket.send_text(token)

            await executor_future
            await websocket.send_text("\n")

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"[backend] websocket handler error: {exc}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)
