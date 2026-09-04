import queue

import numpy as np


class FeatureChunkQueue:
    """Bridges a live queue of raw PCM increments into the chunked mel-feature
    generator that model.generate(input_features=...) consumes.

    Reproduces the nemetron/backend.py prototype's chunking contract
    (processor.num_samples_first_audio_chunk / num_samples_per_audio_chunk,
    is_first_audio_chunk flag) but instead of slicing a complete, pre-known
    numpy array it blocks on a live queue — so a single generate() call can
    span an entire mic-on-to-mic-off recording instead of restarting per
    network message. push()/stop() are called from the WS receive loop
    (any thread); build_first_chunk()/generator() are consumed from the
    background generate-thread and block until enough audio has arrived.
    """

    def __init__(self, processor, model, sample_rate: int, language: str | None = None):
        self._processor = processor
        self._model = model
        self._sample_rate = sample_rate
        self._language = language
        self._queue: "queue.Queue[np.ndarray | None]" = queue.Queue()
        self._buffer = np.empty(0, dtype=np.float32)
        self._ended = False
        self._first_chunk_inputs = None

    def push(self, pcm: np.ndarray) -> None:
        self._queue.put(pcm)

    def stop(self) -> None:
        self._queue.put(None)

    def _fill(self, n_samples: int) -> bool:
        """Grow the buffer until it holds >= n_samples. Returns False if the
        stream ended (stop() consumed) before that many samples arrived."""
        while self._buffer.shape[0] < n_samples:
            if self._ended:
                return False
            item = self._queue.get()
            if item is None:
                self._ended = True
                return False
            self._buffer = np.concatenate([self._buffer, item])
        return True

    def build_first_chunk(self):
        """Blocks until the first chunk's worth of audio is available (padding
        with zeros if the recording ended before that), runs the processor on
        it, and caches + returns the resulting BatchFeature. Must be called
        once, before generator()."""
        processor = self._processor
        model = self._model
        first_chunk_n = processor.num_samples_first_audio_chunk

        if not self._fill(first_chunk_n):
            pad = np.zeros(first_chunk_n - self._buffer.shape[0], dtype=np.float32)
            self._buffer = np.concatenate([self._buffer, pad])

        kwargs = dict(
            sampling_rate=self._sample_rate,
            is_streaming=True,
            is_first_audio_chunk=True,
            return_tensors="pt",
        )
        if self._language:
            kwargs["language"] = self._language

        inputs = processor(self._buffer[:first_chunk_n], **kwargs)
        self._first_chunk_inputs = inputs.to(model.device, dtype=model.dtype)
        return self._first_chunk_inputs

    def generator(self):
        """Yields input_features tensors: the cached first chunk, then
        successive chunks pulled from the live queue until the stream ends."""
        if self._first_chunk_inputs is None:
            raise RuntimeError("build_first_chunk() must be called before generator()")

        processor = self._processor
        model = self._model

        yield self._first_chunk_inputs.input_features[:, : processor.num_mel_frames_first_audio_chunk, :]

        mel_frame_idx = processor.num_mel_frames_first_audio_chunk
        hop_length = processor.feature_extractor.hop_length
        n_fft = processor.feature_extractor.n_fft
        chunk_n = processor.num_samples_per_audio_chunk

        while True:
            start_idx = mel_frame_idx * hop_length - n_fft // 2
            end_idx = start_idx + chunk_n
            have_full_chunk = self._fill(end_idx)

            if not have_full_chunk:
                remaining = self._buffer.shape[0] - start_idx
                if remaining <= 0:
                    return
                pad = np.zeros(chunk_n - remaining, dtype=np.float32)
                self._buffer = np.concatenate([self._buffer, pad])

            inputs = processor(
                self._buffer[start_idx:end_idx],
                sampling_rate=self._sample_rate,
                is_streaming=True,
                is_first_audio_chunk=False,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device, dtype=model.dtype)
            yield inputs.input_features
            mel_frame_idx += processor.num_mel_frames_per_audio_chunk

            if not have_full_chunk:
                return
