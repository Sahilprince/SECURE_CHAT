FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and startup script
COPY main.py main.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8000

# ✅ Run through bash to expand $PORT
CMD ["bash", "-c", "/app/start.sh"]
