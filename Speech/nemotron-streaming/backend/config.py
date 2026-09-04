import torch

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

DEFAULT_LANGUAGE = "en-US"
NUM_LOOKAHEAD_TOKENS = 6

# How long the WS relay loop waits for the next transcript event before
# treating the session as stalled.
EVENT_TIMEOUT_SECONDS = 120.0

# How long to wait for the background generate() thread to join after stop().
JOIN_TIMEOUT_SECONDS = 30.0
