# ruh-serverless

RunPod Serverless worker for the **Ruh English NLP model** (46M params,
khalq_akhar epoch_19). The model is too heavy for the Mizan VPS CPU
(per-token full forward passes froze the server), so it runs here on GPU,
pay-per-second, scale-to-zero.

## How it works

- `handler.py` — loads `RuhModel` + `BayanTokenizer` once at worker startup,
  serves OpenAI-style chat completions via `runpod.serverless`.
- `Dockerfile` — `runpod/pytorch` CUDA base + `runpod` SDK + `ruh_model/` +
  checkpoint (176MB, from the GitHub Release).
- `.github/workflows/build.yml` — builds and pushes
  `ghcr.io/codewithjuber/ruh-serverless:latest` on every relevant push.

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

`model.pt` + `config.json` live on the `checkpoint-epoch19` GitHub Release
(uploaded once; the Dockerfile pulls them at build time). Source of truth for
the weights is the Mizan VPS volume `mizan_mizan-data`
(`ruh-checkpoints/khalq_akhar/epoch_19/`).
