# ============================================================
# Dockerfile  —  Financial Time Series Forecasting & Anomaly Detection
# ============================================================

FROM python:3.11-slim

# System deps: psycopg2-binary cần libpq, torch/numpy cần build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài requirements trước để tận dụng Docker layer cache
COPY requirements.txt .

# Tách CPU-only torch để giảm image size (không cần GPU trong container)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code
COPY . .

# Tạo thư mục logs, outputs (sẽ được mount qua volumes)
RUN mkdir -p logs gbm_output tft_output reports

# Default: chạy scheduler (override trong docker-compose)
CMD ["python", "-m", "data_collection_api.scheduler"]
