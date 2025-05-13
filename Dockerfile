FROM python:3.11-slim

# Sistem paketlerini yükle
RUN apt-get update && apt-get install -y build-essential libpq-dev

# Çalışma dizini
WORKDIR /app

# Gereksinimler
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kodları kopyala
COPY . .

# Varsayılan komut
CMD ["python", "runner.py"]
