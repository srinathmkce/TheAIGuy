export type MicButtonState = 'idle' | 'connecting' | 'recording' | 'finishing' | 'error';

interface MicButtonProps {
  state: MicButtonState;
  onClick: () => void;
}

const LABELS: Record<MicButtonState, string> = {
  idle: 'Start',
  connecting: 'Connecting…',
  recording: 'Stop',
  finishing: 'Finishing…',
  error: 'Retry',
};

export function MicButton({ state, onClick }: MicButtonProps) {
  return (
    <button
      type="button"
      className={`mic-button mic-button--${state}`}
      onClick={onClick}
      disabled={state === 'connecting' || state === 'finishing'}
    >
      <span className="mic-button__icon" aria-hidden="true">
        🎙
      </span>
      {LABELS[state]}
    </button>
  );
}
