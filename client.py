#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import time  # new import
from datetime import datetime

import requests

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
            {"role": "user", "content": question},
        ]
    }
    resp = requests.post(LLM_API_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def call_tts(text: str, out_path: str) -> None:
    resp = requests.post(TTS_API_URL, json={"text": text})
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)


def play_audio(path: str) -> None:
    try:
        subprocess.run(["aplay", path], check=True)
    except FileNotFoundError:
        print(
            "Error: 'aplay' not found. Install alsa-utils or adjust play_audio().",
            file=sys.stderr,
        )
    except subprocess.CalledProcessError as e:
        print(f"Playback failed: {e}", file=sys.stderr)


# --- Main CLI ---
def main():
    parser = argparse.ArgumentParser(
        description="Ask an LLM a question, TTS the answer, and play it."
    )
    parser.add_argument(
        "-q", "--question", required=True, help="The question to send to the LLM"
    )
    parser.add_argument(
        "-m", "--model", default="gpt-3.5-turbo", help="Chat model to use"
    )
    parser.add_argument(
        "-s",
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for the LLM",
    )
    args = parser.parse_args()

    # --- LLM timing ---
    print("🧠 Asking LLM…")
    t0 = time.perf_counter()
    answer = call_llm(args.question, args.model, args.system_prompt)
    t1 = time.perf_counter()
    print(f"⏱️  LLM call took {(t1 - t0):.2f} seconds")
    print("💬 LLM says:", answer)

    # prepare output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_wav = f"answer_{timestamp}.wav"

    # --- TTS timing ---
    print("🔊 Generating speech…")
    t2 = time.perf_counter()
    call_tts(answer, out_wav)
    t3 = time.perf_counter()
    print(f"⏱️  TTS generation took {(t3 - t2):.2f} seconds")
    print(f"👉 Saved audio to {out_wav}")

    # --- Playback timing ---
    print("▶️ Playing back…")
    t4 = time.perf_counter()
    play_audio(out_wav)
    t5 = time.perf_counter()
    print(f"⏱️  Playback took {(t5 - t4):.2f} seconds")

    # --- Cleanup ---
    try:
        os.remove(out_wav)
        print(f"🗑️ Removed temporary file {out_wav}")
    except OSError as e:
        print(f"⚠️ Could not remove {out_wav}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
