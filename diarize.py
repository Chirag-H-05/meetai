from pyannote.audio import Pipeline
import torch
import os

# -----------------------------
# Configuration
# -----------------------------
AUDIO_FILE = "audio/bashinimeet_16k.wav"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load pipeline (OFFLINE)
# -----------------------------
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    cache_dir="models"
)

# Force GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

# -----------------------------
# Run diarization
# -----------------------------
diarization = pipeline(AUDIO_FILE)

# -----------------------------
# Save output (RTTM + TXT)
# -----------------------------
rttm_path = os.path.join(OUTPUT_DIR, "diarization.rttm")
txt_path = os.path.join(OUTPUT_DIR, "diarization.txt")

with open(rttm_path, "w") as rttm_file:
    diarization.write_rttm(rttm_file)

with open(txt_path, "w") as txt_file:
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        line = f"{segment.start:.2f} {segment.end:.2f} {speaker}"
        print(line)
        txt_file.write(line + "\n")

print("\n✅ Diarization completed")
print(f"RTTM saved to: {rttm_path}")
print(f"Text output saved to: {txt_path}")
