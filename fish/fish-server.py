import io
import os

import litserve as ls
import numpy as np
import soundfile as sf
import torch
from fastapi import Response
from tools.api import inference
from tools.commons import ServeTTSRequest
from tools.llama.generate import launch_thread_safe_queue
from tools.vqgan.inference import load_model as load_decoder_model


class FishSpeechLitAPI(ls.LitAPI):
    def setup(self, device):
        # You can override these via env vars if you like
        llama_ckpt = os.environ.get(
            "FISH_LLAMA_CHECKPOINT", "checkpoints/fish-speech-1.5"
        )
        decoder_ckpt = os.environ.get(
            "FISH_DECODER_CHECKPOINT",
            "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
        )
        decoder_conf = os.environ.get("FISH_DECODER_CONFIG", "firefly_gan_vq")

        self.device = device
        precision = torch.bfloat16

        # Launch the LLaMA-based semantic generator queue
        self.llama_queue = launch_thread_safe_queue(
            checkpoint_path=llama_ckpt,
            device=device,
            precision=precision,
            compile=False,
        )
        # Load the VQGAN decoder
        self.decoder_model = load_decoder_model(
            config_name=decoder_conf, checkpoint_path=decoder_ckpt, device=device
        )
        print(f"[FishSpeech] Loaded models on {device}")

    def decode_request(self, request):
        # Expecting JSON: {"text": "..."}
        return request["text"]

    def predict(self, text):
        # Build a ServeTTSRequest (no reference examples)
        req = ServeTTSRequest(
            text=text,
            references=[],
            reference_id=None,
            max_new_tokens=0,
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7,
            emotion=None,
            stream=False,
            format="wav",
        )

        # Inject our models into the tools.api module so inference() picks them up
        import tools.api as api_mod

        api_mod.llama_queue = self.llama_queue
        api_mod.decoder_model = self.decoder_model

        # Run Fish-Speech inference (non-streaming)
        gen = inference(req)
        wav = next(gen)  # numpy array of shape (n_samples,)
        return wav

    def encode_response(self, prediction):
        # prediction is a numpy waveform array
        buf = io.BytesIO()
        sf.write(
            buf,
            prediction,
            samplerate=self.decoder_model.spec_transform.sample_rate,
            format="WAV",
        )
        buf.seek(0)
        return Response(content=buf.read(), headers={"Content-Type": "audio/wav"})


if __name__ == "__main__":
    api = FishSpeechLitAPI()
    server = ls.LitServer(api)
    server.run(port=int(os.environ.get("PORT", 5000)))
