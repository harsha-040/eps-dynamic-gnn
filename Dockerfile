FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch: this app only runs inference on 5-node graphs (no
# training, no GPU needed), so the CUDA build would just be dead weight.
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY src ./src
COPY app ./app
COPY configs ./configs
COPY data/processed/metadata.json ./data/processed/metadata.json
COPY results/checkpoints ./results/checkpoints
COPY results/metrics ./results/metrics

EXPOSE 8000
# Shell form so $PORT expands: Render (and most PaaS hosts) inject PORT at
# runtime and expect the app to bind to it; falls back to 8000 locally.
CMD uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
