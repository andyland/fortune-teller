FROM dustynv/torchaudio:2.6.0-r36.4.0-cu128

WORKDIR /app

RUN git clone https://github.com/myshell-ai/MeloTTS.git && \
	cd MeloTTS && \
	pip install -e . && \
	python3 -m unidic download

RUN python3 -m nltk.downloader averaged_perceptron_tagger_eng
RUN pip install litserve

ENV HF_HOME=/checkpoints/hf \
    HF_HUB_CACHE=/checkpoints/hf/hub \
    TRANSFORMERS_CACHE=/checkpoints/hf/hub


COPY melo-server.py .

CMD ["python3", "melo-server.py"]
