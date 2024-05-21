import sounddevice as sd
import numpy as np
import wave

# Parameters for recording
FORMAT = 'int16'           # 16-bit resolution
CHANNELS = 1               # Mono channel
RATE = 44100               # Sampling rate (samples per second)
RECORD_SECONDS = 1         # Duration of recording
OUTPUT_FILENAME = "output.wav"  # Output file name

print("Recording...")

# Record audio
recording = sd.rec(int(RECORD_SECONDS * RATE), samplerate=RATE, channels=CHANNELS, dtype=FORMAT)
sd.wait()  # Wait until recording is finished

print("Finished recording.")

# Save the recorded data as a WAV file
recording = np.squeeze(recording)  # Remove single-dimensional entries from the shape
wf = wave.open(OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(np.dtype(FORMAT).itemsize)
wf.setframerate(RATE)
wf.writeframes(recording.tobytes())
wf.close()

print(f"Saved to {OUTPUT_FILENAME}")
