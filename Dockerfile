FROM python:3.10-slim

# Sätt arbetskatalog
WORKDIR /app

# Installera systemberoenden (för pandas, numpy etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Kopiera requirements.txt och installera
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiera resten av koden
COPY . .

# Starta boten via FastAPI + Uvicorn (håller Render vid liv)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
