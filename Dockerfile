FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for DuckDB and ONNX
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install the package
RUN pip install --no-cache-dir .

# Create a volume for global cache (models and config)
VOLUME /root/.cache/code-rag

# Default command
ENTRYPOINT ["code-rag"]
CMD ["--help"]
