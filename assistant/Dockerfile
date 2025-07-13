FROM dustynv/torchaudio:2.6.0-r36.4.0-cu128

WORKDIR /app

# Install necessary packages
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    build-essential \
    gcc \
    alsa-utils \
    libasound2 \
    libasound2-plugins \
    pulseaudio \
    pulseaudio-utils \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /run/user/1000/pulse
RUN chmod 777 /run/user/1000/pulse

# Install Python dependencies
RUN pip install --no-cache-dir \
    requests \
    soundfile \
    numpy \
    pyaudio \
    sounddevice

RUN pip install --no-cache-dir silero-vad

COPY . .

CMD python assistant.py --microphone 0,0
