from dataclasses import dataclass
from typing import Literal

EventKind = Literal["ready", "partial", "final", "error"]


@dataclass
class TranscriptEvent:
    kind: EventKind
    text: str = ""
    message: str = ""
