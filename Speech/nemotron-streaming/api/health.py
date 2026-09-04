from fastapi import APIRouter

from backend import config
from backend.model_registry import is_loaded

router = APIRouter()


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": is_loaded(),
        "model_id": config.MODEL_ID,
        "device": config.DEVICE,
    }
