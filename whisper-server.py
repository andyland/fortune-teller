import os
import time
from fastapi import UploadFile
import litserve as ls
from whisper_trt import load_trt_model

# Environment-configurable settings
MODEL_NAME = os.getenv("WHISPER_MODEL", "base.en")
PORT = int(os.getenv("PORT", "6001"))

# Load TensorRT-optimized Whisper

def load_model():
    return load_trt_model(MODEL_NAME)

class WhisperTRTLitAPI(ls.LitAPI):
    def setup(self, device):
        print(f"Initializing Whisper TRT model '{MODEL_NAME}' on {device}...")
        self.model = load_model()
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
        # Return only the transcription text
        return {"transcription": output.get("text", "").strip()}

if __name__ == "__main__":
    api = WhisperTRTLitAPI()
    server = ls.LitServer(
        api
    )
    server.run(port=PORT)

