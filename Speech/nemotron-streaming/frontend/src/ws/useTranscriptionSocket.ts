import { useCallback, useRef, useState } from 'react';
import type { ServerMessage } from '../types';

export type SocketState = 'idle' | 'connecting' | 'open' | 'closed' | 'error';

const WS_URL = `ws://${window.location.hostname}:8000/ws/transcribe`;
const SAMPLE_RATE = 16000;
const LANGUAGE = 'en-US';

export function useTranscriptionSocket(onMessage: (message: ServerMessage) => void) {
  const [state, setState] = useState<SocketState>('idle');
  const socketRef = useRef<WebSocket | null>(null);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  const connect = useCallback(() => {
    return new Promise<void>((resolve, reject) => {
      setState('connecting');
      const socket = new WebSocket(WS_URL);
      socket.binaryType = 'arraybuffer';
      socketRef.current = socket;

      socket.onopen = () => {
        socket.send(JSON.stringify({ type: 'start', sampleRate: SAMPLE_RATE, language: LANGUAGE }));
      };

      socket.onmessage = (event: MessageEvent<string>) => {
        const message = JSON.parse(event.data) as ServerMessage;
        if (message.type === 'ready') {
          setState('open');
          resolve();
        }
        onMessageRef.current(message);
      };

      socket.onerror = () => {
        setState('error');
        reject(new Error('WebSocket connection failed — is the backend running on port 8000?'));
      };

      socket.onclose = () => {
        setState((prev) => (prev === 'error' ? prev : 'closed'));
        socketRef.current = null;
      };
    });
  }, []);

  const sendAudioChunk = useCallback((pcm: Float32Array) => {
    const socket = socketRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(pcm.buffer as ArrayBuffer);
    }
  }, []);

  const sendStop = useCallback(() => {
    const socket = socketRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: 'stop' }));
    }
  }, []);

  const disconnect = useCallback(() => {
    socketRef.current?.close();
    socketRef.current = null;
  }, []);

  return { state, connect, sendAudioChunk, sendStop, disconnect };
}
