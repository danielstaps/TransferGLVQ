# Reproducible environment for T-GLVQ (TensorFlow / protoflow).
# CPU image; for GPU use a matching tensorflow-gpu / CUDA base instead.
# Pinning policy: original paper-time versions where known, otherwise the newest
# versions verified to run (see repo-cleanup MANIFEST). Deps loosely pinned below.
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

#   docker run --rm -it <image> python3 -m examples.bonbons
CMD ["python", "-c", "import tensorflow, numpy, matplotlib; print('T-GLVQ env OK')"]
