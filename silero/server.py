import io, os, time
import torch
import soundfile as sf
from fastapi import UploadFile
import litserve as ls

class VADAPI(ls.LitAPI):
    def setup(self, device):
        """Load the Silero VAD model."""
        # Load the pre-trained Silero VAD model and its utilities from torch.hub
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        # utils returns a tuple: (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)
        (self.get_speech_timestamps, _, self.read_audio, _, _) = utils
        # Define the sampling rate for inference (Silero VAD supports 8000 and 16000 Hz)
        self.sampling_rate = 16000

    def decode_request(self, request):
        audio_bytes = request["content"].file.read()

        os.makedirs("tmp", exist_ok=True)
        path = f"tmp/{time.time()}.wav"
        with open(path, "wb") as f:
            f.write(audio_bytes)
        return path

    def predict(self, path):
        # Decode audio bytes into a numpy array and sample rate
        data = self.read_audio(path)
        # Run VAD to get timestamps of speech segments
        speech_timestamps = self.get_speech_timestamps(data, self.model, return_seconds=True)
        print(speech_timestamps)
        # Determine if any speech was detected
        has_voice = len(speech_timestamps) > 0
        return {"has_voice": has_voice}

if __name__ == "__main__":
    # Instantiate the API with a batch size of 1
    api = VADAPI(max_batch_size=1)
    # Create and run the LitServe server on port 6004
    server = ls.LitServer(api, accelerator="auto")
    server.run(port=6004)

