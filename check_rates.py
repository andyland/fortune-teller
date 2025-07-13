import sounddevice as sd

device = 0  # USB mic index

for rate in [8000, 16000, 22050, 24000, 32000, 44100, 48000]:
    try:
        sd.check_input_settings(device=device, samplerate=rate)
        print(f"✅ Works: {rate} Hz")
    except Exception as e:
        print(f"❌ Fails: {rate} Hz – {e}")
