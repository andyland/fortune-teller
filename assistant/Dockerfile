FROM python:3.10-slim

WORKDIR /app

# Install necessary packages
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    build-essential \
    gcc \
    alsa-utils \
    libasound2 \
    libasound2-plugins \
    && rm -rf /var/lib/apt/lists/*

# Configure ALSA
RUN echo "pcm.!default {\ntype hw\ncard 2\ndevice 0\n}" > /etc/asound.conf
RUN echo "ctl.!default {\ntype hw\ncard 2\ndevice 0\n}" >> /etc/asound.conf

# Create necessary ALSA directories
RUN mkdir -p /var/run/alsa
RUN chown -R root:audio /var/run/alsa
RUN chmod 775 /var/run/alsa

# Install Python dependencies
RUN pip install --no-cache-dir \
    requests \
    soundfile \
    numpy \
    pyaudio \
    sounddevice

COPY . .

CMD python assistant.py
