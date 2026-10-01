# ruh-serverless

RunPod Serverless worker for the **experimental Ruh root model** (46M params,
khalq_akhar epoch_19). The model is too heavy for the Mizan VPS CPU
(per-token full forward passes froze the server), so it runs here on GPU,
pay-per-second, scale-to-zero.

## How it works

- `handler.py` — loads `RuhModel` + `BayanTokenizer` once at worker startup,
  serves OpenAI-style chat completions via `runpod.serverless`.
- `Dockerfile` — digest-pinned Python runtime, PyTorch 2.5.1 CUDA 12.4 wheels,
  RunPod SDK, source code, and explicitly verified epoch-19 weights.
- `.github/workflows/build.yml` — builds and pushes
  immutable commit tags plus `latest` after CPU model and worker contract tests.

## Request format

```json
{
  "input": {
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 256,
    "temperature": 1.0
  }
}
```

## Response format

```json
{
  "choices": [{"message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 3, "completion_tokens": 42, "total_tokens": 45},
  "model": "ruh-khalq-akhar-epoch19",
  "elapsed_s": 4.2
}
```

## Checkpoint

The final weights are pinned to Kaggle dataset `zubairshaikh/ruh-model-checkpoints`, version 1, `checkpoints/khalq_akhar/epoch_19`. The weight file is 183,991,059 bytes; the larger dataset archive contains other epochs. `checkpoint-manifest.json` records exact sizes, SHA-256 digests, and tokenizer version 1.

```bash
# KAGGLE_API_TOKEN may be supplied through environment settings when required.
python scripts/fetch_checkpoint.py
python scripts/fetch_checkpoint.py --verify
docker build -t ruh-serverless:local .
```

The runtime base is mirrored byte-for-byte from Docker Official Images and pinned by digest. BuildKit's optional `build_ca` secret accepts a trusted CA bundle for environments with a TLS proxy; it is never copied into the image. CUDA execution requires a GPU runtime. A CPU-only diagnostic build can use `--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu` and `RUH_DEVICE=cpu`.

Epoch 19 remains Bayan v1 with 62 mapped IDs. It was trained on synthetic root fragments; it is not a validated English chat model. Inference keeps all conversation turns, removes terminal prompt EOS, masks unmapped IDs, preserves temperature zero, and returns only generated tokens. Fixing those contracts does not make the weights learn sentences. New v2 training preserves surface text and needs a separately trained, evaluated checkpoint. Copy `vocab.json` and `tokenizer.json` with every new checkpoint and update the immutable manifest; never relabel legacy weights as v2.
