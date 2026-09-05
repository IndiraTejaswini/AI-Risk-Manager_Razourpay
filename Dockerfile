FROM python:3.13-slim AS base

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY data/ ./data/
COPY features/ ./features/
COPY models/ ./models/
COPY policy/ ./policy/
COPY responder/ ./responder/
COPY api/ ./api/
COPY eval/ ./eval/

ENV OMP_NUM_THREADS=4

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health').read()"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
