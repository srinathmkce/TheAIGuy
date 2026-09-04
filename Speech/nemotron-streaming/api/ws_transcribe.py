import asyncio
import json

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend import config
from backend.model_registry import get_model, get_processor
from backend.session import StreamingSession

from .schemas import ErrorMessage, FinalMessage, PartialMessage, ReadyMessage, StartMessage

router = APIRouter()

# Guards the single shared GPU model: only one streaming session may run
# model.generate() at a time. A connection that arrives while the lock is
# held is rejected immediately with a "busy" error rather than queued.
_session_lock = asyncio.Lock()


@router.websocket("/ws/transcribe")
async def transcribe_ws(websocket: WebSocket) -> None:
    await websocket.accept()

    if _session_lock.locked():
        await websocket.send_json(ErrorMessage(message="Another transcription session is already active.").model_dump())
        await _safe_close(websocket)
        return

    async with _session_lock:
        await _run_session(websocket)


async def _run_session(websocket: WebSocket) -> None:
    loop = asyncio.get_running_loop()

    start_msg = await _receive_start(websocket)
    if start_msg is None:
        await _safe_close(websocket)
        return

    session = StreamingSession(get_model(), get_processor(), loop, language=start_msg.language)
    session.start()
    await websocket.send_json(ReadyMessage().model_dump())

    relay_task = asyncio.create_task(_relay_events(websocket, session))

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data is not None:
                session.push_audio(np.frombuffer(data, dtype=np.float32))
                continue

            text = message.get("text")
            if text is None:
                continue
            payload = _parse_json(text)
            if payload is not None and payload.get("type") == "stop":
                break
    except WebSocketDisconnect:
        pass
    finally:
        session.stop()
        await loop.run_in_executor(None, session.join, config.JOIN_TIMEOUT_SECONDS)
        await relay_task
        await _safe_close(websocket)


async def _receive_start(websocket: WebSocket) -> StartMessage | None:
    while True:
        try:
            message = await websocket.receive()
        except WebSocketDisconnect:
            return None
        if message["type"] == "websocket.disconnect":
            return None
        text = message.get("text")
        if text is None:
            continue
        payload = _parse_json(text)
        if payload is not None and payload.get("type") == "start":
            return StartMessage(**payload)


async def _relay_events(websocket: WebSocket, session: StreamingSession) -> None:
    async for event in session.events():
        try:
            if event.kind == "partial":
                await websocket.send_json(PartialMessage(text=event.text).model_dump())
            elif event.kind == "final":
                await websocket.send_json(FinalMessage(text=event.text).model_dump())
            elif event.kind == "error":
                await websocket.send_json(ErrorMessage(message=event.message).model_dump())
        except Exception:
            break


def _parse_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


async def _safe_close(websocket: WebSocket) -> None:
    try:
        await websocket.close()
    except RuntimeError:
        pass
