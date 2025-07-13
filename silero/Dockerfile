FROM dustynv/torchaudio:2.6.0-r36.4.0-cu128

WORKDIR /app

RUN pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 litserve python-multipart
RUN pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-cache-dir silero-vad soundfile librosa

COPY server.py .

CMD ["python3", "server.py"]
