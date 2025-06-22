import argparse
import requests
from datetime import datetime

# Update this URL to your server's URL if hosted remotely
API_URL = "http://127.0.0.1:6000/predict"

def send_request(text):

    response = requests.post(API_URL, json={"text": text})
    if response.status_code == 200:
        filename = "output.wav"
        
        with open(filename, "wb") as audio_file:
            audio_file.write(response.content)
        
        print(f"Audio saved to {filename}")
    else:
        print(f"Error: Response with status code {response.status_code} - {response.text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sends a transcript to a F5 text to speech server")
    parser.add_argument("--text", required=True, help="Text transcript of the speech to generate")
    args = parser.parse_args()
    
    send_request(args.text)
