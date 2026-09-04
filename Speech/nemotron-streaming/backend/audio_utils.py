import io

import numpy as np
import soundfile as sf


def decode_audio_file(raw_bytes: bytes, target_sr: int) -> np.ndarray:
    """Decode an audio file's bytes (mp3/wav/etc.) into mono float32 PCM at target_sr.

    Used by the offline smoke-test script and any future file-upload path — the
    live mic path already sends raw float32 PCM at target_sr, so no decoding is
    needed there.
    """
    audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio
