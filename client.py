#!/usr/bin/env python3

import argparse
import requests
import subprocess
from datetime import datetime
import sys

# --- Configuration ---
LLM_API_URL = "http://localhost:8000/v1/chat/completions"
TTS_API_URL = "http://127.0.0.1:6000/predict"
# You can override this via --system-prompt
DEFAULT_SYSTEM_PROMPT = "You are a sarcastic fortune teller at Burning Man.  When given a question, reply with a single, witty, sarcastic fortune."

# --- Functions ---
def call_llm(question: str, model: str, system_prompt: str) -> str:
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question}
        ]
    }
    resp = requests.post(LLM_API_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()
    # assume the first choice is what we want
    return data["choices"][0]["message"]["content"]

def call_tts(text: str, out_path: str) -> None:
    resp = requests.post(TTS_API_URL, json={"text": text})
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)

def play_audio(path: str) -> None:
    # uses the ALSA 'aplay' utility; adjust if you need another player
    try:
        subprocess.run(["aplay", path], check=True)
    except FileNotFoundError:
        print("Error: 'aplay' not found. Install alsa-utils or adjust play_audio().", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Playback failed: {e}", file=sys.stderr)

# --- Main CLI ---
def main():
    parser = argparse.ArgumentParser(
        description="Ask an LLM a question, TTS the answer, and play it."
    )
    parser.add_argument(
        "-q", "--question",
        required=True,
        help="The question to send to the LLM"
    )
    parser.add_argument(
        "-m", "--model",
        default="gpt-3.5-turbo",
        help="Chat model to use (default: %(default)s)"
    )
    parser.add_argument(
        "-s", "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for the LLM (default: mystical fortune teller)"
    )
    args = parser.parse_args()

    print("🧠 Asking LLM…")
    answer = call_llm(args.question, args.model, args.system_prompt)
    print("💬 LLM says:", answer)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_wav = f"answer_{timestamp}.wav"

    print("🔊 Generating speech…")
    call_tts(answer, out_wav)
    print(f"👉 Saved audio to {out_wav}")

    print("▶️ Playing back…")
    play_audio(out_wav)

if __name__ == "__main__":
    main()
