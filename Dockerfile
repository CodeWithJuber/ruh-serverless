# Rebuildable CUDA runtime plus explicit verified weights; no inherited worker output.
FROM mirror.gcr.io/library/python:3.11-slim-bookworm@sha256:a36c24f9cbdf4fd0f52d67f0823eeac19c2028c637cecc392d97f980d4fec56b
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ENV PYTHONUNBUFFERED=1 RUH_CHECKPOINT_DIR=/app/checkpoint RUH_DEVICE=cuda
WORKDIR /app
COPY checkpoint-manifest.json /app/checkpoint-manifest.json
COPY scripts/fetch_checkpoint.py /app/scripts/fetch_checkpoint.py
COPY checkpoint/ /app/checkpoint/
COPY handler.py /app/handler.py
COPY backend/ /app/backend/
COPY ruh_model/ /app/ruh_model/
RUN --mount=type=secret,id=build_ca \
    if [ -f /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca; fi; \
    python scripts/fetch_checkpoint.py --verify \
    && pip install --no-cache-dir "torch==2.5.1" --index-url "$TORCH_INDEX_URL" \
    && pip install --no-cache-dir "runpod==1.7.10" "numpy==1.26.4"
CMD ["python", "/app/handler.py"]
