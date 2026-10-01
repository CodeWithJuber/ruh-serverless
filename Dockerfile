# Ruh English NLP model (46M) — RunPod serverless worker.
# Base already carries Python 3.11 + PyTorch with CUDA; we add the runpod SDK,
# the ruh_model package, the checkpoint, and the handler.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    RUH_CHECKPOINT_DIR=/app/checkpoint \
    RUH_DEVICE=cuda

RUN pip install --no-cache-dir runpod

WORKDIR /app
COPY ruh_model/ /app/ruh_model/

# Checkpoint baked in at build time (downloaded from the GitHub Release).
ARG CHECKPOINT_URL
RUN mkdir -p /app/checkpoint && \
    curl -sSL -o /app/checkpoint/model.pt "${CHECKPOINT_URL}/model.pt" && \
    curl -sSL -o /app/checkpoint/config.json "${CHECKPOINT_URL}/config.json" && \
    ls -la /app/checkpoint/

COPY handler.py /app/

CMD ["python", "/app/handler.py"]
