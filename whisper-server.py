import re
import os
import time
from fastapi import UploadFile
import litserve as ls

MODEL_NAME = os.getenv("WHISPER_MODEL", "base.en")
PORT = int(os.getenv("PORT", "6001"))

class WhisperTRTLitAPI(ls.LitAPI):
    def setup(self, device):
        print(f"Initializing Whisper TRT model '{MODEL_NAME}' on {device}...")
        from whisper_trt import load_trt_model
        self.model = load_trt_model(MODEL_NAME)
        print("Whisper TRT setup complete.")

    def decode_request(self, request):
        audio_bytes = request["content"].file.read()
        
        os.makedirs("tmp", exist_ok=True)
        path = f"tmp/{time.time()}.wav"
        with open(path, "wb") as f:
            f.write(audio_bytes)
        return path

    def predict(self, audio_path):
        # Transcribe and clean up
        result = self.model.transcribe(audio_path)
        os.remove(audio_path)
        return result

    def encode_response(self, output):
        # 1) Get raw Whisper text
        text = output.get("text", "")

        # 2) Remove annotations in […] or (…)
        text = re.sub(
            r'[\[\(]\s*[^)\]]*?\s*[\]\)]',  # anything from '[' or '(' up to matching ']' or ')'
            '',
            text,
            flags=re.IGNORECASE
        )

        # 3) Remove any leftover bracket or parenthesis characters
        text = re.sub(r'[\[\]\(\)]', '', text)

        # 4) Collapse multiple spaces/newlines into one space
        text = re.sub(r'\s+', ' ', text).strip()

        return {"transcription": text}

if __name__ == "__main__":
    api = WhisperTRTLitAPI()
    server = ls.LitServer(
        api
    )
    server.run(port=PORT)

