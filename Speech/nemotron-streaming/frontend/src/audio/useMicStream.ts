import { useCallback, useRef, useState } from 'react';

const TARGET_SAMPLE_RATE = 16000;
// Served from public/ as a plain static file — see that file's header comment
// for why the worklet processor isn't part of the TS/Vite bundle graph.
const PCM_WORKLET_URL = '/pcm-worklet-processor.js';

export type MicState = 'idle' | 'starting' | 'recording' | 'error';

export function useMicStream() {
  const [state, setState] = useState<MicState>('idle');
  const [error, setError] = useState<string | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);

  const stop = useCallback(() => {
    workletNodeRef.current?.disconnect();
    workletNodeRef.current = null;

    audioContextRef.current?.close().catch(() => {});
    audioContextRef.current = null;

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    setState((prev) => (prev === 'error' ? prev : 'idle'));
  }, []);

  const start = useCallback(
    async (onChunk: (pcm: Float32Array) => void) => {
      setError(null);
      setState('starting');
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
          video: false,
        });
        streamRef.current = stream;

        const audioContext = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
        await audioContext.resume();
        audioContextRef.current = audioContext;

        await audioContext.audioWorklet.addModule(PCM_WORKLET_URL);

        const source = audioContext.createMediaStreamSource(stream);
        const workletNode = new AudioWorkletNode(audioContext, 'pcm-worklet-processor');
        workletNode.port.onmessage = (event: MessageEvent<Float32Array>) => {
          onChunk(event.data);
        };
        source.connect(workletNode);
        // AudioWorkletNode only gets pulled by the audio graph when reachable
        // from the destination; the processor never writes an output so this
        // stays silent (no mic feedback).
        workletNode.connect(audioContext.destination);
        workletNodeRef.current = workletNode;

        setState('recording');
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
        setState('error');
        stop();
        throw err;
      }
    },
    [stop],
  );

  return { state, error, start, stop };
}
