#!/usr/bin/env python3
import argparse
import io
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf


def find_microphone_device(device_name_pattern="CVL-2005"):
    """Find microphone device by name pattern"""
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            device_info = str(device)
            # Look for device with the pattern and input channels
            if (device_name_pattern.lower() in device_info.lower() 
                and device.get('max_input_channels', 0) > 0):
                print(f"🎤 Auto-detected microphone: Device {i} - {device['name']}")
                return i
        
        # Fallback: find any USB Audio device with input
        for i, device in enumerate(devices):
            device_info = str(device)
            if ("usb audio" in device_info.lower() 
                and device.get('max_input_channels', 0) > 0):
                print(f"🎤 Found USB Audio device: Device {i} - {device['name']}")
                return i
                
        print("⚠️ No CVL-2005 or USB Audio device found, using default")
        return None
    except Exception as e:
        print(f"❌ Error detecting microphone: {e}")
        return None


class VoiceAssistant:
    def __init__(
        self,
        whisper_url,
        llm_url,
        tts_url,
        vad_url="http://localhost:6004/predict",
        microphone="CVL-2005",
        samplerate=16000,
        channels=1,
        buffer_duration=20.0,
        blocksize=1024,
        system_prompt=None,
    ):
        self.whisper_url = whisper_url
        self.llm_url = llm_url
        self.tts_url = tts_url
        self.vad_url = vad_url
        
        # Auto-detect microphone if needed
        if microphone == "auto" or microphone == "CVL-2005":
            detected_mic = find_microphone_device()
            self.microphone = detected_mic if detected_mic is not None else 27  # fallback to default
        else:
            try:
                self.microphone = int(microphone)
            except (ValueError, TypeError):
                self.microphone = microphone
        
        self.samplerate = samplerate
        self.channels = channels
        self.buffer_duration = buffer_duration
        self.blocksize = blocksize
        self.system_prompt = (
            system_prompt
            or "You are a helpful AI assistant. Provide clear, concise responses."
        )

        # Calculate rolling buffer size
        chunks_needed = int(np.ceil(buffer_duration * samplerate / blocksize))
        self.audio_buffer = deque(maxlen=chunks_needed)
        self.buffer_lock = threading.Lock()

        # State management
        self.last_transcription = ""
        self.last_transcription_time = 0
        self.silence_threshold = 3.0  # seconds of same transcription = silence
        self.vad_silence_threshold = 1.0  # seconds without voice activity
        self.vad_buffer_duration = 2.0  # seconds of audio for VAD analysis
        self.last_voice_activity_time = 0
        self.speech_start_time = 0
        self.is_speaking = False
        self.processing = False
        self.paused = False  # Flag to pause audio processing
        self.stream = None

        # Conversation history for context
        self.conversation_history = []
        self.max_history_length = 10  # Keep last 10 exchanges
        self.last_activity_time = time.time()  # Track last question time
        self.history_timeout = 60.0  # Clear history after 60s of inactivity

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"InputStream status: {status}", file=sys.stderr)

        # Skip recording if we're paused
        if self.paused:
            return

        chunk = indata.copy()
        with self.buffer_lock:
            self.audio_buffer.append(chunk)

    def get_transcription(self):
        """Get transcription from current audio buffer"""
        if self.paused:
            return None

        with self.buffer_lock:
            if not self.audio_buffer:
                return None
            audio_data = np.concatenate(list(self.audio_buffer), axis=0)

        # Write to BytesIO as WAV
        wav_io = io.BytesIO()
        sf.write(wav_io, audio_data, self.samplerate, format="WAV")
        wav_io.seek(0)

        files = {"content": ("buffer.wav", wav_io, "audio/wav")}
        try:
            resp = requests.post(self.whisper_url, files=files, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            transcription = data.get("transcription") or data.get("text", "")
            return transcription.strip()
        except requests.RequestException as e:
            print(f"Whisper error: {e}", file=sys.stderr)
            return None

    def call_llm(self, question: str) -> str:
        """Call LLM with the question and conversation context"""
        # Update activity time
        self.last_activity_time = time.time()

        # Build messages with conversation history
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history
        for exchange in self.conversation_history:
            messages.append({"role": "user", "content": exchange["question"]})
            messages.append({"role": "assistant", "content": exchange["answer"]})

        # Add current question
        messages.append({"role": "user", "content": question})

        payload = {"messages": messages}

        try:
            resp = requests.post(self.llm_url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]

            # Store this exchange in history
            self.conversation_history.append(
                {"question": question, "answer": answer, "timestamp": time.time()}
            )

            # Trim history if it gets too long
            if len(self.conversation_history) > self.max_history_length:
                self.conversation_history.pop(0)

            return answer

        except requests.RequestException as e:
            print(f"LLM error: {e}", file=sys.stderr)
            return "Sorry, I couldn't process your request."

    def call_tts(self, text: str, out_path: str) -> bool:
        """Generate TTS audio file"""
        try:
            resp = requests.post(self.tts_url, json={"text": text}, timeout=30)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(resp.content)
            return True
        except requests.RequestException as e:
            print(f"TTS error: {e}", file=sys.stderr)
            return False

    def check_voice_activity(self) -> bool:
        """Check if there's current voice activity using Silero VAD"""
        if not self.vad_url or self.paused:
            return False

        with self.buffer_lock:
            if not self.audio_buffer:
                return False
            
            # Use only the last N seconds for VAD (more responsive)
            vad_chunks_needed = int(np.ceil(self.vad_buffer_duration * self.samplerate / self.blocksize))
            recent_chunks = list(self.audio_buffer)[-vad_chunks_needed:]
            if not recent_chunks:
                return False
            audio_data = np.concatenate(recent_chunks, axis=0)

        # Write to BytesIO as WAV
        wav_io = io.BytesIO()
        sf.write(wav_io, audio_data, self.samplerate, format="WAV")
        wav_io.seek(0)

        files = {"content": ("buffer.wav", wav_io, "audio/wav")}
        try:
            resp = requests.post(self.vad_url, files=files, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            has_voice = data.get("has_voice", False)
            
            # Update speech state based on VAD
            current_time = time.time()
            if has_voice:
                self.last_voice_activity_time = current_time
                if not self.is_speaking:
                    # Speech just started
                    self.is_speaking = True
                    self.speech_start_time = current_time
                    print(f"\r🎤 Speech started", end="", flush=True)
                else:
                    print(f"\r🔊 Speaking... ({current_time - self.speech_start_time:.1f}s)", end="", flush=True)
            else:
                if self.is_speaking:
                    silence_duration = current_time - self.last_voice_activity_time if self.last_voice_activity_time > 0 else 0
                    print(f"\r🔇 Speech paused (silent for {silence_duration:.1f}s)", end="", flush=True)
                else:
                    print(f"\r👂 Listening...", end="", flush=True)
            
            return has_voice
        except requests.RequestException as e:
            print(f"VAD error: {e}", file=sys.stderr)
            return False

    def check_history_timeout(self):
        """Clear conversation history if inactive for too long"""
        current_time = time.time()
        if (
            self.conversation_history
            and current_time - self.last_activity_time > self.history_timeout
        ):
            history_count = len(self.conversation_history)
            self.conversation_history.clear()
            print(
                f"\n🕐 Cleared {history_count} conversation exchange(s) after {self.history_timeout:.0f}s of inactivity"
            )
            print("👂 Starting fresh conversation...")
            return True
        return False

    def play_audio(self, path: str) -> None:
        """Play audio file"""
        try:
            subprocess.run(
                ["aplay", "-D", "plughw:1,0", path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print("Error: 'aplay' not found. Install alsa-utils.", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Playback failed: {e}", file=sys.stderr)

    def process_question(self, question: str):
        """Process complete question through LLM and TTS"""
        if self.processing:
            return

        self.processing = True
        self.paused = True  # Pause audio processing to free GPU
        print(f"\n🎤 Question detected: '{question}'")
        print("⏸️ Pausing audio processing to free GPU resources...")

        try:
            # Clear audio buffer completely - start fresh afterward
            with self.buffer_lock:
                self.audio_buffer.clear()

            # Call LLM
            print("🧠 Thinking...")
            t0 = time.perf_counter()
            answer = self.call_llm(question)
            t1 = time.perf_counter()
            print(f"💭 Response ({(t1-t0):.1f}s): {answer}")

            # Generate TTS
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            with tempfile.TemporaryDirectory() as tmpdir:
                out_wav = f"{tmpdir}/response_{timestamp}.wav"

                print("🔊 Generating speech...")
                t2 = time.perf_counter()
                if self.call_tts(answer, out_wav):
                    t3 = time.perf_counter()
                    print(f"▶️ Playing response ({(t3-t2):.1f}s generation)...")

                    t4 = time.perf_counter()
                    self.play_audio(out_wav)
                    t5 = time.perf_counter()
                    print(f"✅ Done ({(t5-t4):.1f}s playback)")

        except Exception as e:
            print(f"❌ Error processing question: {e}", file=sys.stderr)
        finally:
            # Clear buffer again and reset all state for fresh start
            with self.buffer_lock:
                self.audio_buffer.clear()

            self.processing = False
            self.paused = False  # Resume audio processing

            # Reset transcription state completely
            self.last_transcription = ""
            self.last_transcription_time = time.time()
            self.last_voice_activity_time = 0
            self.speech_start_time = 0
            self.is_speaking = False

            print("🗑️ Cleared audio buffer for fresh recording")
            print("▶️ Resuming audio processing...")

            # Show conversation count for context awareness
            if len(self.conversation_history) > 0:
                print(
                    f"💬 Conversation context: {len(self.conversation_history)} exchange(s) remembered"
                )

            print("👂 Listening for next question...")

    def monitor_transcriptions(self):
        """Monitor VAD and transcriptions to detect completed questions"""
        while True:
            try:
                # Check for history timeout during idle periods
                if not self.paused and not self.processing:
                    self.check_history_timeout()

                # Skip processing if we're handling a question
                if self.paused:
                    time.sleep(1)
                    continue

                current_time = time.time()
                
                # Always check VAD to track speech state
                has_voice = self.check_voice_activity()
                
                # Only get transcription if we're speaking or just finished speaking
                transcription = None
                if self.is_speaking or (self.last_voice_activity_time > 0 and 
                                       current_time - self.last_voice_activity_time < self.vad_silence_threshold * 2):
                    transcription = self.get_transcription()

                # Update transcription display if we have speech
                if transcription and len(transcription) > 5:
                    if transcription != self.last_transcription:
                        self.last_transcription = transcription
                        self.last_transcription_time = current_time
                        speech_duration = current_time - self.speech_start_time if self.speech_start_time > 0 else 0
                        print(
                            f"\r🎙️ ({speech_duration:.1f}s): {transcription[:50]}{'...' if len(transcription) > 50 else ''}",
                            end="",
                            flush=True,
                        )

                # Check for end of speech using VAD
                if (
                    self.is_speaking  # We were speaking
                    and not has_voice  # But VAD says no voice now
                    and self.last_voice_activity_time > 0
                    and current_time - self.last_voice_activity_time > self.vad_silence_threshold
                    and not self.processing
                    and transcription
                    and len(transcription.strip()) > 10
                ):
                    # Speech has ended, process the question
                    self.is_speaking = False
                    print("\r" + " " * 80 + "\r", end="")

                    # Process the question in a separate thread to avoid blocking
                    question_thread = threading.Thread(
                        target=self.process_question, args=(transcription,)
                    )
                    question_thread.daemon = True
                    question_thread.start()

                    # Wait for processing to start before continuing
                    time.sleep(0.5)

                time.sleep(0.2)  # Check every 200ms for responsive VAD

            except Exception as e:
                print(f"\nMonitoring error: {e}", file=sys.stderr)
                time.sleep(1)

    def start(self):
        """Start the voice assistant"""
        try:
            self.stream = sd.InputStream(
                device=self.microphone,
                samplerate=self.samplerate,
                channels=self.channels,
                blocksize=self.blocksize,
                callback=self.audio_callback,
            )
            self.stream.start()
            print("🎤 Voice Assistant started!")
            print("👂 Listening for questions... (speak for 3+ seconds, then pause)")
            print("Press Ctrl+C to stop.\n")

            # Start monitoring in a separate thread
            monitor_thread = threading.Thread(target=self.monitor_transcriptions)
            monitor_thread.daemon = True
            monitor_thread.start()

            # Keep main thread alive
            while True:
                time.sleep(0.1)

        except Exception as e:
            print(f"Error starting audio stream: {e}", file=sys.stderr)
            return False

    def stop(self):
        """Stop the voice assistant"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("\n🛑 Voice Assistant stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Voice-activated AI assistant with automatic question detection"
    )

    # Whisper settings
    parser.add_argument(
        "--whisper-url",
        default="http://localhost:6001/predict",
        help="Whisper TRT endpoint URL (default: %(default)s)",
    )

    # LLM settings
    parser.add_argument(
        "--llm-url",
        default="http://localhost:8000/v1/chat/completions",
        help="LLM API endpoint URL (default: %(default)s)",
    )

    # TTS settings
    parser.add_argument(
        "--tts-url",
        default="http://127.0.0.1:6000/predict",
        help="TTS API endpoint URL (default: %(default)s)",
    )

    # VAD settings
    parser.add_argument(
        "--vad-url",
        default="http://localhost:6004/predict",
        help="Silero VAD API endpoint URL (default: %(default)s)",
    )

    # Audio settings
    parser.add_argument(
        "--samplerate",
        type=int,
        default=16000,
        help="Sampling rate in Hz (default: %(default)s)",
    )
    parser.add_argument(
        "--microphone",
        default="CVL-2005",
        help="Microphone device to use (device number, 'auto', or 'CVL-2005' for auto-detect)",
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
        help="Audio buffer duration in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=1024,
        help="Audio blocksize (default: %(default)s)",
    )

    # AI settings
    parser.add_argument(
        "--system-prompt",
        default="You are a sarcastic fortune teller at Burning Man.  When given a question, reply with a single, witty, sarcastic fortune.",
        help="System prompt for the LLM",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=2.0,
        help="Seconds of silence before processing question (default: %(default)s)",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=10,
        help="Maximum conversation exchanges to remember (default: %(default)s)",
    )
    parser.add_argument(
        "--history-timeout",
        type=float,
        default=60.0,
        help="Clear conversation history after N seconds of inactivity (default: %(default)s)",
    )
    parser.add_argument(
        "--vad-silence-threshold",
        type=float,
        default=1.0,
        help="Seconds of VAD silence before processing question (default: %(default)s)",
    )

    args = parser.parse_args()

    # Create and start assistant
    assistant = VoiceAssistant(
        whisper_url=args.whisper_url,
        llm_url=args.llm_url,
        tts_url=args.tts_url,
        vad_url=args.vad_url,
        microphone=args.microphone,
        samplerate=args.samplerate,
        channels=args.channels,
        buffer_duration=args.buffer_duration,
        blocksize=args.blocksize,
        system_prompt=args.system_prompt,
    )

    assistant.silence_threshold = args.silence_threshold
    assistant.max_history_length = args.max_history
    assistant.history_timeout = args.history_timeout
    assistant.vad_silence_threshold = args.vad_silence_threshold

    try:
        assistant.start()
    except KeyboardInterrupt:
        assistant.stop()
        sys.exit(0)


if __name__ == "__main__":
    main()
