import asyncio
import traceback
from threading import Thread
from typing import AsyncIterator

import numpy as np
from transformers import TextIteratorStreamer

from . import config
from .events import TranscriptEvent
from .feature_stream import FeatureChunkQueue


class StreamingSession:
    """One continuous streaming-transcription session, spanning a single
    recording from mic-on to mic-off. Owns exactly one background thread
    running one persistent model.generate() call for the session's lifetime —
    this is what keeps ASR context alive across the whole recording instead
    of restarting per network message."""

    def __init__(self, model, processor, loop: asyncio.AbstractEventLoop, language: str = config.DEFAULT_LANGUAGE):
        self._model = model
        self._processor = processor
        self._loop = loop
        self._sample_rate = processor.feature_extractor.sampling_rate
        self._feature_queue = FeatureChunkQueue(processor, model, self._sample_rate, language)
        self._events: asyncio.Queue = asyncio.Queue()
        self._thread: Thread | None = None

    def start(self) -> None:
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def push_audio(self, pcm_f32: np.ndarray) -> None:
        self._feature_queue.push(pcm_f32)

    def stop(self) -> None:
        self._feature_queue.stop()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    def _emit(self, event: TranscriptEvent | None) -> None:
        self._loop.call_soon_threadsafe(self._events.put_nowait, event)

    def _run(self) -> None:
        try:
            first_chunk_inputs = self._feature_queue.build_first_chunk()
            streamer = TextIteratorStreamer(self._processor.tokenizer, skip_special_tokens=True)
            gen_kwargs = {
                **first_chunk_inputs,
                "input_features": self._feature_queue.generator(),
                "streamer": streamer,
            }
            generate_thread = Thread(target=self._model.generate, kwargs=gen_kwargs, daemon=True)
            generate_thread.start()

            for token in streamer:
                if token:
                    self._emit(TranscriptEvent(kind="partial", text=token))

            generate_thread.join()
            self._emit(TranscriptEvent(kind="final"))
        except Exception as exc:
            traceback.print_exc()
            self._emit(TranscriptEvent(kind="error", message=str(exc)))
        finally:
            self._emit(None)
