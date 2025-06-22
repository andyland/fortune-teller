FROM python:3.10-slim

WORKDIR /app

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    requests \
    soundfile \
    numpy \
    pyaudio \
    sounddevice

COPY . .

CMD python assistant.py 2>&1
