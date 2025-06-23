import base64
import io
import os
import time

import litserve as ls
import numpy as np
import soundfile as sf
import torch
from f5_tts import api
from fastapi import Response

SPEAKER_WAV_FILE = "obama.wav"
LANGUAGE = "en"


class F5TTSLitAPI(ls.LitAPI):
    def setup(self, device):
        print("using {}".format(device))
        self.f5tts = api.F5TTS(
            device=device, model="E2TTS_Base", hf_cache_dir="/checkpoints"
        )
        print("setup complete...")

    def decode_request(self, request):
        return request["text"]

    def predict(self, text):
        wav, s, t = self.f5tts.infer(
            ref_file=str(SPEAKER_WAV_FILE),
            ref_text="I want to thank his counterparts on the other side",
            gen_text=text,
            seed=-1,  # random seed = -1
        )

        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, wav, samplerate=22050, format="WAV")
        audio_buffer.seek(0)
        audio_data = audio_buffer.getvalue()
        audio_buffer.close()

        return audio_data

    def encode_response(self, prediction):
        return Response(content=prediction, headers={"Content-Type": "audio/wav"})


if __name__ == "__main__":
    api = F5TTSLitAPI()
    server = ls.LitServer(api)
    server.run(port=7000)
