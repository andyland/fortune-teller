FROM python:3.10-slim

WORKDIR /app

# Install necessary packages
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    build-essential \
    gcc \
    pulseaudio \
    pulseaudio-utils \
    libasound2 \
    libasound2-plugins \
    && rm -rf /var/lib/apt/lists/*

# Set up PulseAudio
RUN mkdir -p /var/run/pulse
RUN chown -R pulse:pulse /var/run/pulse

# Set environment variables
ENV PULSE_SERVER=unix:/run/pulse/native \
    PULSE_COOKIE=/run/pulse/cookie \
    PULSE_RUNTIME_PATH=/run/user/1000/pulse

# Create PulseAudio runtime directory
RUN mkdir -p /run/user/1000/pulse
RUN chown -R 1000:1000 /run/user/1000/pulse

# Install Python dependencies
RUN pip install --no-cache-dir \
    requests \
    soundfile \
    numpy \
    pyaudio \
    sounddevice

COPY . .

CMD python assistant.py
