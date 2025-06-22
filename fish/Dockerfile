FROM dustynv/torchaudio:2.6.0-r36.4.0-cu128

# 1) Install system packages needed by Fish-Speech
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsox-dev ffmpeg \
        libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 \
        build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Clone Fish-Speech and install in editable "stable" mode
RUN git clone https://github.com/fishaudio/fish-speech.git \
 && cd fish-speech \
 && pip install --no-cache-dir -e . \
 && cd .. \
 && ln -s fish-speech/tools tools


ENV HF_HOME=/checkpoints/hf \
	HF_HUB_CACHE=/checkpoints/hf/hub \
	TRANSFORMERS_CACHE=/checkpoints/hf/hub \
	XDG_CACHE_HOME=/checkpoints/hf

# 3) Install litserve (and any other Python deps you need)
RUN pip install --no-cache-dir litserve

# 4) Copy your server script + reference WAV
COPY fish-server.py obama.wav ./

# 5) Launch on port 7000
CMD ["python3", "fish-server.py"]
