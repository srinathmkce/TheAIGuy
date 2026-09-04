from threading import Lock

from transformers import AutoModelForRNNT, AutoProcessor

from . import config

_processor: AutoProcessor | None = None
_model: AutoModelForRNNT | None = None
_load_lock = Lock()


def load_model() -> None:
    """Load the processor/model once. Safe to call more than once (idempotent)."""
    global _processor, _model
    with _load_lock:
        if _model is not None:
            return
        print(f"[model_registry] loading {config.MODEL_ID} on {config.DEVICE} (dtype={config.DTYPE})")
        _processor = AutoProcessor.from_pretrained(config.MODEL_ID)
        _processor.set_num_lookahead_tokens(config.NUM_LOOKAHEAD_TOKENS)
        _model = AutoModelForRNNT.from_pretrained(
            config.MODEL_ID, torch_dtype=config.DTYPE, device_map=config.DEVICE
        )
        print("[model_registry] model ready")


def get_processor() -> AutoProcessor:
    if _processor is None:
        raise RuntimeError("Model not loaded yet — call load_model() first.")
    return _processor


def get_model() -> AutoModelForRNNT:
    if _model is None:
        raise RuntimeError("Model not loaded yet — call load_model() first.")
    return _model


def is_loaded() -> bool:
    return _model is not None
