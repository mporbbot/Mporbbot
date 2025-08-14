FROM python:3.10-slim

# Sätt arbetskatalog
WORKDIR /app

# Installera systemberoenden (för pandas, numpy etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Kopiera alla filer
COPY . .

# Installera Python-paket
RUN pip install --no-cache-dir \
    requests \
    python-telegram-bot \
    pandas \
    numpy \
    python-dotenv \
    kucoin-python \
    fastapi \
    uvicorn \
    httpx

# Starta boten med Uvicorn (FastAPI håller Render-processen vid liv)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
