"""RunPod serverless handler for the Ruh English NLP model (46M params).

Loads the model once at worker startup (warm), then serves OpenAI-style
chat-completion requests:

    input:  {"messages": [{"role": "user", "content": "..."}], "max_tokens": 256, "temperature": 1.0}
    output: {"choices": [{"message": {"role": "assistant", "content": "..."}}], "usage": {...}}

The generation mirrors backend/providers_ruh.py::_extract_prompt/_tokens_to_tensors
so the VPS side needs only a thin HTTP adapter, not a rewrite.
"""
import logging
import os
import time

import runpod
import torch

from ruh_model.model import RuhModel
from ruh_model.tokenizer.bayan import BayanTokenizer

logger = logging.getLogger("ruh-handler")

CHECKPOINT_DIR = os.environ.get("RUH_CHECKPOINT_DIR", "/app/checkpoint")
DEVICE = os.environ.get("RUH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_NEW_TOKENS_CAP = int(os.environ.get("RUH_MAX_TOKENS_CAP", "512"))

logger.info("Loading Ruh model from %s on %s ...", CHECKPOINT_DIR, DEVICE)
print(f"[ruh] Loading model from {CHECKPOINT_DIR} on {DEVICE} ...", flush=True)
try:
    _model = RuhModel.from_pretrained(CHECKPOINT_DIR)
    _model.to(DEVICE)
    _model.train(False)  # inference mode
    _tokenizer = BayanTokenizer()
    print("[ruh] Model loaded and warm.", flush=True)
except Exception as e:
    print(f"[ruh] FATAL: Failed to load model: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
    raise
logger.info("Ruh model loaded and warm.")


def _extract_prompt(messages: list) -> str:
    """Get the text of the last user message (mirrors providers_ruh)."""
    for msg in reversed(messages or []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            texts = [b.get("text", "") for b in content if b.get("type") == "text"]
            return " ".join(texts)
        return str(content)
    return ""


def _generate(prompt: str, max_tokens: int, temperature: float) -> tuple[str, int, int]:
    tokens = _tokenizer.encode(prompt)
    if not tokens:
        return "", 0, 0
    root_ids = torch.tensor([[t[0] for t in tokens]], dtype=torch.long, device=DEVICE)
    pattern_ids = torch.tensor([[t[1] for t in tokens]], dtype=torch.long, device=DEVICE)
    max_new = max(1, min(max_tokens or 256, MAX_NEW_TOKENS_CAP))
    with torch.no_grad():
        generated = _model.generate(
            root_ids,
            pattern_ids,
            max_new_tokens=max_new,
            temperature=temperature if temperature else 1.0,
        )
    gen_list = generated[0].tolist() if generated.ndim == 2 else generated.tolist()
    gen_tokens = [(int(rid), 0) for rid in gen_list]
    text = _tokenizer.decode(gen_tokens)
    return text, len(tokens), len(gen_list)


def handler(job: dict) -> dict:
    """RunPod handler: job["input"] carries the OpenAI-style request."""
    try:
        inp = job.get("input", {}) or {}
        messages = inp.get("messages", [])
        max_tokens = inp.get("max_tokens", 256)
        temperature = inp.get("temperature", 1.0)
        prompt = _extract_prompt(messages)
        if not prompt.strip():
            return {"error": "no user message found in input.messages"}
        started = time.time()
        text, in_tok, out_tok = _generate(prompt, max_tokens, temperature)
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": in_tok,
                "completion_tokens": out_tok,
                "total_tokens": in_tok + out_tok,
            },
            "model": "ruh-khalq-akhar-epoch19",
            "elapsed_s": round(time.time() - started, 2),
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("generation failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


runpod.serverless.start({"handler": handler})
