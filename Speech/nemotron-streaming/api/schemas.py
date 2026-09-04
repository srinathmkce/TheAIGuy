from typing import Literal

from pydantic import BaseModel

from backend import config


class StartMessage(BaseModel):
    type: Literal["start"] = "start"
    sampleRate: int = 16000
    language: str = config.DEFAULT_LANGUAGE


class StopMessage(BaseModel):
    type: Literal["stop"] = "stop"


class ReadyMessage(BaseModel):
    type: Literal["ready"] = "ready"


class PartialMessage(BaseModel):
    type: Literal["partial"] = "partial"
    text: str


class FinalMessage(BaseModel):
    type: Literal["final"] = "final"
    text: str = ""


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    message: str
