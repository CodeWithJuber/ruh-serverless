"""RunPod serverless handler for the Ruh English NLP model (46M params).

Loads the model once at worker startup (warm), then serves OpenAI-style
chat-completion requests:

    input:  {"messages": [{"role": "user", "content": "..."}], "max_tokens": 256, "temperature": 1.0}
    output: {"choices": [{"message": {"role": "assistant", "content": "..."}}], "usage": {...}}

The generation mirrors backend/providers_ruh.py::_extract_prompt/_tokens_to_tensors
so the VPS side needs only a thin HTTP adapter, not a rewrite.
"""
import math
import logging
import threading
import os
import time

import runpod
import torch

from ruh_model.model import RuhModel
from ruh_model.tokenizer.bayan import BayanTokenizer
from ruh_model.tokenizer.conversation import serialize_messages

logger = logging.getLogger("ruh-handler")

CHECKPOINT_DIR = os.environ.get("RUH_CHECKPOINT_DIR", "/app/checkpoint")
DEVICE = os.environ.get("RUH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_NEW_TOKENS_CAP = int(os.environ.get("RUH_MAX_TOKENS_CAP", "512"))

_model = None
_tokenizer = None
_load_lock = threading.Lock()


def load_model():
    global _model, _tokenizer
    with _load_lock:
        if _model is None:
            logger.info("Loading Ruh checkpoint on %s", DEVICE)
            model = RuhModel.from_pretrained(CHECKPOINT_DIR).to(DEVICE)
            model.eval()
            _tokenizer = model.tokenizer or BayanTokenizer.from_pretrained(CHECKPOINT_DIR)
            _model = model


def _extract_prompt(messages: list) -> str:
    return serialize_messages(messages)


def _generate(prompt: str, max_tokens: int, temperature: float) -> tuple[str, int, int]:
    load_model()
    tokens = _tokenizer.encode(prompt, add_eos=False)
    tokens = tokens[-_model.config.max_seq_len:]
    if not tokens:
        return "", 0, 0
    root_ids = torch.tensor([[t[0] for t in tokens]], dtype=torch.long, device=DEVICE)
    pattern_ids = torch.tensor([[t[1] for t in tokens]], dtype=torch.long, device=DEVICE)
    max_new = max(1, min(max_tokens or 256, MAX_NEW_TOKENS_CAP))
    # Valid vocab size: model trained with n_roots=4000 embedding but only
    # 62 IDs are valid (4 special + 58 real roots). Mask the rest.
    valid_n_roots = _tokenizer._vocab.n_roots
    with torch.no_grad():
        generated = _model.generate(
            root_ids,
            pattern_ids,
            max_new_tokens=max_new,
            temperature=1.0 if temperature is None else temperature,
            valid_n_roots=valid_n_roots,
        )
    gen_list = generated[0].tolist() if generated.ndim == 2 else generated.tolist()
    # Only decode the newly generated tokens (skip the prompt)
    # Model uses default_pattern_id=1 for generated tokens
    new_tokens = gen_list[len(tokens):]
    gen_tokens = [(int(rid), 1) for rid in new_tokens]
    text = _tokenizer.decode(gen_tokens)
    return text, len(tokens), len(new_tokens)


def handler(job: dict) -> dict:
    """RunPod handler: job["input"] carries the OpenAI-style request."""
    try:
        inp = job.get("input", {}) or {}
        messages = inp.get("messages", [])
        if not isinstance(messages, list) or not messages or len(messages) > 100 or any(not isinstance(message, dict) for message in messages):
            raise ValueError("messages must contain 1..100 message objects")
        if len(str(messages)) > 200000:
            raise ValueError("Conversation exceeds size limit")
        max_tokens = inp.get("max_tokens", 256)
        temperature = inp.get("temperature", 1.0)
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if temperature is not None and (not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0):
            raise ValueError("temperature must be finite and nonnegative")
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


if __name__ == "__main__":
    load_model()
    runpod.serverless.start({"handler": handler})
