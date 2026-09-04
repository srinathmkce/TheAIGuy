import { useCallback, useRef, useState } from 'react';
import './App.css';
import { MicButton, type MicButtonState } from './components/MicButton';
import { StatusIndicator } from './components/StatusIndicator';
import { TranscriptView } from './components/TranscriptView';
import { useMicStream } from './audio/useMicStream';
import { useTranscriptionSocket } from './ws/useTranscriptionSocket';
import type { ServerMessage } from './types';

const STATUS_LABEL: Record<MicButtonState, string> = {
  idle: 'Ready — click the microphone to start.',
  connecting: 'Connecting…',
  recording: 'Recording & transcribing in real time…',
  finishing: 'Finishing up…',
  error: 'Something went wrong.',
};

function App() {
  const [appState, setAppState] = useState<MicButtonState>('idle');
  const [transcript, setTranscript] = useState('');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const micRef = useRef<ReturnType<typeof useMicStream> | null>(null);

  const handleServerMessage = useCallback((message: ServerMessage) => {
    if (message.type === 'partial') {
      setTranscript((prev) => prev + message.text);
    } else if (message.type === 'final') {
      setTranscript((prev) => prev + message.text);
      setAppState('idle');
    } else if (message.type === 'error') {
      setErrorMessage(message.message);
      setAppState('error');
      micRef.current?.stop();
    }
  }, []);

  const socket = useTranscriptionSocket(handleServerMessage);
  const mic = useMicStream();
  micRef.current = mic;

  const handleMicClick = useCallback(async () => {
    if (appState === 'recording') {
      setAppState('finishing');
      socket.sendStop();
      mic.stop();
      return;
    }
    if (appState === 'connecting' || appState === 'finishing') {
      return;
    }

    setErrorMessage(null);
    setTranscript('');
    setAppState('connecting');
    try {
      await socket.connect();
      await mic.start(socket.sendAudioChunk);
      setAppState('recording');
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : String(err));
      setAppState('error');
      mic.stop();
      socket.disconnect();
    }
  }, [appState, mic, socket]);

  const handleClear = useCallback(() => {
    setTranscript('');
  }, []);

  const handleDownload = useCallback(() => {
    if (!transcript.trim()) return;
    const blob = new Blob([transcript], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'transcript.txt';
    link.click();
    URL.revokeObjectURL(url);
  }, [transcript]);

  const active = appState === 'recording' || appState === 'connecting' || appState === 'finishing';

  return (
    <div className="app">
      <header className="app__header">
        <h1>Streaming Transcription</h1>
        <p className="app__subtitle">Powered by nvidia/nemotron-3.5-asr-streaming-0.6b</p>
      </header>

      <div className="controls">
        <MicButton state={appState} onClick={handleMicClick} />
        <button type="button" className="btn btn-secondary" onClick={handleClear}>
          Clear
        </button>
        <button type="button" className="btn btn-secondary" onClick={handleDownload}>
          Download
        </button>
      </div>

      <StatusIndicator label={errorMessage ?? STATUS_LABEL[appState]} active={active} isError={appState === 'error'} />

      <TranscriptView text={transcript} active={active} />
    </div>
  );
}

export default App;
