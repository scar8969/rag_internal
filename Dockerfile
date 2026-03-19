FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY internal/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY internal/rag/ ./rag/
COPY internal/__init__.py ./rag/__init__.py
COPY main.py ./

RUN mkdir -p /app/data /app/data/chroma

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py", "--help"]
