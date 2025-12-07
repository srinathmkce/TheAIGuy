FROM python:3.10-slim

# Install system dependencies (espeak-ng is crucial for Kokoro)
RUN apt-get update && apt-get install -y \
    espeak-ng \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only Torch (keeps image light)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Kokoro and Web API dependencies
RUN pip install --no-cache-dir kokoro>=0.19 soundfile fastapi uvicorn python-multipart

# Copy the API script
COPY api.py .

# Expose the port we will use
EXPOSE 8080

# Start the API server
ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]