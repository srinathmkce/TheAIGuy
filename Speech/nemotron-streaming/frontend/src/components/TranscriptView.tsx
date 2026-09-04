interface TranscriptViewProps {
  text: string;
  active: boolean;
}

export function TranscriptView({ text, active }: TranscriptViewProps) {
  return (
    <div className="transcript-box">
      {text}
      {active && <span className="cursor" aria-hidden="true" />}
    </div>
  );
}
