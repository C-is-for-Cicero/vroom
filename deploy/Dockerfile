FROM python:3.11-slim

WORKDIR /app

# LightGBM runtime dependency
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[web]"

ENV F1PRED_DATA_DIR=/data
VOLUME /data

EXPOSE 8000
CMD ["uvicorn", "vroom.app:app", "--host", "0.0.0.0", "--port", "8000"]
