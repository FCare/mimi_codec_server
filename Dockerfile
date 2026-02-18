FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Download Mimi model using hf CLI
RUN hf download kyutai/mimi --local-dir /app/model && \
    echo "Model downloaded to /app/model"

# Copy server code
COPY server.py ./

# Expose port
EXPOSE 6542

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:6542/health')"

# Run the server
CMD ["python", "server.py"]
