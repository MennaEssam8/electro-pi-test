FROM python:3.11-slim

WORKDIR /app

# System deps for torch (CPU-only) build wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch wheel -- much smaller download than the default CUDA build,
# and this container has no GPU access anyway.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000

# Single worker: the model is loaded once into this process's memory at
# startup (see app.py lifespan). Multiple uvicorn workers would each load
# their own copy of the model, multiplying memory use -- see README for how
# this changes for real concurrent-user scale.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
