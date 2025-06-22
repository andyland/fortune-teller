#!/usr/bin/env python3
import argparse
import requests
import sys

def transcribe(file_path: str, url: str):
    try:
        with open(file_path, 'rb') as f:
            files = {'content': f}
            resp = requests.post(url, files=files)
        resp.raise_for_status()
    except FileNotFoundError:
        print(f"Error: file not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    except requests.RequestException as e:
        print(f"Request error: {e}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    # The Whisper TRT server returns a dict with a "text" key
    text = data.get("transcription")
    if text is None:
        print("No 'text' field in response:", data, file=sys.stderr)
        sys.exit(1)

    print(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send an audio file to Whisper TRT LitServe and print the transcription"
    )
    parser.add_argument(
        "file",
        help="Path to the audio file to transcribe (e.g. speech.wav)"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:6001/predict",
        help="Whisper TRT endpoint URL (default: %(default)s)"
    )
    args = parser.parse_args()
    transcribe(args.file, args.url)
