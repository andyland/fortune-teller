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
    TRANSFORMERS_CACHE=/checkpoints/hf/hub \
    XDG_CACHE_HOME=/checkpoints/hf \
    CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    PATH=$PATH:/usr/local/cuda/bin \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# Verify CUDA and GPU availability
RUN nvidia-smi && \
    python3 -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); print(f'Current device: {torch.cuda.current_device()}'); print(f'Device count: {torch.cuda.device_count()}')"

# Copy your reference WAV and the new server script
COPY melo-server.py .

# Launch the MeloTTS LitServe app
CMD ["python3", "melo-server.py"]
