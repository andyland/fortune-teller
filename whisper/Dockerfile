FROM dustynv/whisper_trt:r36.3.0

RUN pip install python-multipart
RUN pip install git+https://github.com/Lightning-AI/LitServe.git@main

COPY whisper-server.py .

CMD ["python3", "whisper-server.py"]
