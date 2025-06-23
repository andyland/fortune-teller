import io
import os
import tempfile
import time

import litserve as ls
from fastapi import Response

SPEAKER_LANGUAGE = "EN"  # base language code
DEFAULT_SPEAKER = "EN-US"  # default accent
SAMPLE_RATE = 22050  # MeloTTS uses 22050 Hz by default
SPEED = 1.0  # adjustable speed


class MeloTTSLitAPI(ls.LitAPI):
    def setup(self, device):
        from melo.api import TTS

        print(f"Initializing MeloTTS on {device}...")
        # initialize the TTS model
        self.tts = TTS(language=SPEAKER_LANGUAGE, device=device)
        # pick the default speaker ID
        spk2id = self.tts.hps.data.spk2id
        if DEFAULT_SPEAKER not in spk2id:
            raise ValueError(f"Speaker '{DEFAULT_SPEAKER}' not found in MeloTTS model")
        self.speaker_id = spk2id[DEFAULT_SPEAKER]
        print("MeloTTS setup complete.")

    def decode_request(self, request):
        return request["text"]

    def predict(self, text):
        # write to a temp file, then read back bytes
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name

        # synthesize to file
        self.tts.tts_to_file(text, self.speaker_id, out_path, speed=SPEED)

        # read back into memory
        with open(out_path, "rb") as f:
            audio_data = f.read()
        # clean up
        os.remove(out_path)

        return audio_data

    def encode_response(self, prediction):
        return Response(content=prediction, headers={"Content-Type": "audio/wav"})


if __name__ == "__main__":
    api = MeloTTSLitAPI()
    server = ls.LitServer(api, accelerator="cuda")
    server.run(port=6000)
