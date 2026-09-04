export type ServerMessage =
  | { type: 'ready' }
  | { type: 'partial'; text: string }
  | { type: 'final'; text: string }
  | { type: 'error'; message: string };

export interface StartMessage {
  type: 'start';
  sampleRate: number;
  language: string;
}

export interface StopMessage {
  type: 'stop';
}
