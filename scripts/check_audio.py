import soundfile as sf
import os
from glob import glob

# Find the most recent audio file
audio_files = glob('data/audiovis/complex/*.wav')
latest_file = max(audio_files, key=os.path.getctime)

# Load and print audio properties
data, samplerate = sf.read(latest_file)
duration = len(data) / samplerate

print(f"Audio file: {latest_file}")
print(f"Sample rate: {samplerate} Hz")
print(f"Duration: {duration:.2f} seconds")
print(f"Channels: {data.shape[1] if len(data.shape) > 1 else 1}")
print(f"Number of samples: {len(data)}") 