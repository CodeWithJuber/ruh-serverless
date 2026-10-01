# Ruh English NLP model (46M) — RunPod serverless worker.
# Incremental build on top of the previous image (which already has the
# PyTorch base, ruh_model package, and the baked-in checkpoint).
# This layer only pins the runpod SDK, ensures numpy, and updates the handler.
FROM ghcr.io/codewithjuber/ruh-serverless:latest

ENV PYTHONUNBUFFERED=1 \
    RUH_CHECKPOINT_DIR=/app/checkpoint \
    RUH_DEVICE=cuda

RUN pip install --no-cache-dir "runpod==1.7.10" numpy

WORKDIR /app
COPY handler.py /app/handler.py
COPY backend/qca/ /app/backend/qca/

CMD ["python", "/app/handler.py"]
