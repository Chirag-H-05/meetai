from faster_whisper import WhisperModel
from pathlib import Path

AUDIO_FILE = "audio/bashinimeet_16k.wav"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="float32"   # ✅ NO cuBLASLt required
)

segments, info = model.transcribe(
    AUDIO_FILE,
    language="en",
    task="transcribe",
    beam_size=10,
    vad_filter=True,
    vad_parameters=dict(
        min_silence_duration_ms=500
    )
)

out_txt = OUTPUT_DIR / "transcript.txt"

with open(out_txt, "w", encoding="utf-8") as f:
    for seg in segments:
        line = f"[{seg.start:7.2f} → {seg.end:7.2f}] {seg.text.strip()}"
        print(line)
        f.write(line + "\n")

print("\n✅ Transcription finished")
print("Detected language:", info.language)
print("Saved to:", out_txt)
