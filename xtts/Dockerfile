FROM dustynv/xtts:r36.3.0

RUN pip install litserve


# redirect both cache and data into /app
ENV XDG_CACHE_HOME=/app/.cache
ENV XDG_DATA_HOME=/app/.local/share

# create the two tts dirs in your app folder
RUN mkdir -p ${XDG_CACHE_HOME}/tts \
 && mkdir -p ${XDG_DATA_HOME}/tts

WORKDIR /app
COPY tts-server.py .
COPY obama.wav .

CMD ["python3", "tts-server.py"]
