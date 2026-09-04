interface StatusIndicatorProps {
  label: string;
  active: boolean;
  isError?: boolean;
}

export function StatusIndicator({ label, active, isError }: StatusIndicatorProps) {
  return (
    <div className={`status${isError ? ' status--error' : ''}`}>
      {active && <span className="dot" aria-hidden="true" />}
      {label}
    </div>
  );
}
