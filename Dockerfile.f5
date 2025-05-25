FROM dustynv/torchaudio:2.6.0-r36.4.0-cu128

WORKDIR /app
RUN git clone https://github.com/SWivid/F5-TTS.git && cd F5-TTS && pip install -e .  
RUN pip install litserve

COPY obama.wav .
COPY f5-server.py .

CMD ["python3", "f5-server.py"]
