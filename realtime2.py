#!/usr/bin/env python3
import argparse
import io
import sys
import threading
import time
from collections import deque

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(
        description="Continuously capture the last 20s of mic audio and send to Whisper TRT every second."
    )
    parser.add_argument(
        "--url",
        default="http://localhost:6001/predict",
        help="Whisper TRT endpoint URL (default: %(default)s)",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=16000,
        help="Sampling rate in Hz (default: %(default)s)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of audio channels (default: %(default)s)",
    )
    parser.add_argument(
        "--buffer-duration",
        type=float,
        default=20.0,
        help="How many seconds to keep in the rolling buffer (default: %(default)s)",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=1024,
        help="Number of frames per audio callback (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Device to use (default: %(default)s)",
    )
    args = parser.parse_args()

    url = args.url
    samplerate = args.samplerate
    channels = args.channels
    buffer_duration = args.buffer_duration
    blocksize = args.blocksize
    device = args.device

    # Calculate how many chunks of size `blocksize` we need to cover buffer_duration
    chunks_needed = int(np.ceil(buffer_duration * samplerate / blocksize))

    # Rolling buffer: each entry is a NumPy array of shape (blocksize, channels)
    audio_buffer = deque(maxlen=chunks_needed)
    buffer_lock = threading.Lock()

    def audio_callback(indata, frames, time_info, status):
        if status:
            # Print any over/underflows onto stderr
            print(f"InputStream status: {status}", file=sys.stderr)
        # indata is shape (frames, channels)
        chunk = indata.copy()
        with buffer_lock:
            audio_buffer.append(chunk)

    def send_buffer_periodically():
        """
        This function is scheduled to run once per second. It grabs
        the entire rolling buffer, concatenates it, writes to an in‐memory WAV,
        and POSTs to the Whisper TRT endpoint.
        """
        threading.Timer(1.0, send_buffer_periodically).start()

        with buffer_lock:
            if not audio_buffer:
                return
            # Concatenate all chunks into one NumPy array of shape (total_frames, channels)
            audio_data = np.concatenate(list(audio_buffer), axis=0)

        # Write to a BytesIO as a WAV file
        wav_io = io.BytesIO()
        sf.write(wav_io, audio_data, samplerate, format="WAV")
        wav_io.seek(0)

        files = {"content": ("buffer.wav", wav_io, "audio/wav")}
        try:
            resp = requests.post(url, files=files)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} – Request error: {e}",
                file=sys.stderr,
            )
            return

        try:
            data = resp.json()
        except ValueError:
            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} – Failed to parse JSON: {resp.text}",
                file=sys.stderr,
            )
            return

        transcription = data.get("transcription")
        if transcription is None:
            # Some servers use `"text"` instead of `"transcription"`. Try fallback:
            transcription = data.get("text")
        if transcription is not None:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} – {transcription}")
        else:
            print(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} – No 'transcription' field in response: {data}",
                file=sys.stderr,
            )

    # Open an InputStream that calls `audio_callback` for each block
    try:
        stream = sd.InputStream(
            samplerate=samplerate,
            device=device,
            channels=channels,
            blocksize=blocksize,
            callback=audio_callback,
        )
        stream.start()
    except Exception as e:
        print(f"Error opening audio stream: {e}", file=sys.stderr)
        sys.exit(1)

    # Kick off the periodic sender
    send_buffer_periodically()

    print("Recording... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
        stream.stop()
        stream.close()
        sys.exit(0)


if __name__ == "__main__":
    main()
