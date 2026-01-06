from pathlib import Path
import re
import sys
import soundfile as sf
import numpy as np
from collections import defaultdict

from speaker_recognizer import identify_speaker

BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"

AUDIO_FILE = "audio/bashinimeet_16k.wav"
TRANSCRIPT_FILE = OUTPUTS_DIR / "transcript.txt"
RTTM_FILE = OUTPUTS_DIR / "diarization.rttm"
OUTPUT_FILE = OUTPUTS_DIR / "speaker_transcript.txt"

SAMPLE_RATE = 16000


def fail(msg):
    print(f"\n❌ ERROR: {msg}\n")
    sys.exit(1)


# ------------------------------------------------------------
# LOAD RTTM
# ------------------------------------------------------------
def load_rttm(path: Path):
    segments = defaultdict(list)

    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 8:
                continue

            start = float(p[3])
            dur = float(p[4])
            speaker = p[7]
            segments[speaker].append((start, start + dur))

    if not segments:
        fail("No diarization segments found")

    return segments


# ------------------------------------------------------------
# LOAD TRANSCRIPT
# ------------------------------------------------------------
def load_transcript(path: Path):
    pattern = re.compile(
        r"\[\s*(\d+\.?\d*)\s*(?:→|->|–|-)\s*(\d+\.?\d*)\s*\]\s*(.+)"
    )

    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                chunks.append({
                    "start": float(m.group(1)),
                    "end": float(m.group(2)),
                    "text": m.group(3)
                })

    if not chunks:
        fail("No transcript segments parsed")

    return chunks


# ------------------------------------------------------------
# IDENTIFY EACH DIARIZED SPEAKER
# ------------------------------------------------------------
def build_speaker_name_map(audio, diar_segments):
    speaker_map = {}

    for spk, times in diar_segments.items():
        samples = []

        for start, end in times:
            s = int(start * SAMPLE_RATE)
            e = int(end * SAMPLE_RATE)
            samples.append(audio[s:e])

            if sum(len(x) for x in samples) > SAMPLE_RATE * 15:
                break

        clip = np.concatenate(samples)
        name, score = identify_speaker(clip)
        speaker_map[spk] = name

        print(f"🎤 {spk} → {name} (score={score:.3f})")

    return speaker_map


# ------------------------------------------------------------
# FIND SPEAKER BY OVERLAP
# ------------------------------------------------------------
def find_speaker(start, end, diar_segments):
    best_spk = "UNKNOWN"
    best_overlap = 0.0

    for spk, segs in diar_segments.items():
        for s, e in segs:
            overlap = min(end, e) - max(start, s)
            if overlap > best_overlap and overlap > 0:
                best_overlap = overlap
                best_spk = spk

    return best_spk


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    diar_segments = load_rttm(RTTM_FILE)
    transcript = load_transcript(TRANSCRIPT_FILE)

    audio, sr = sf.read(AUDIO_FILE)
    if sr != SAMPLE_RATE:
        fail("Audio must be 16kHz")

    speaker_name_map = build_speaker_name_map(audio, diar_segments)

    lines = []
    for t in transcript:
        spk_id = find_speaker(t["start"], t["end"], diar_segments)
        spk_name = speaker_name_map.get(spk_id, spk_id)

        lines.append(
            f"[{t['start']:.2f}–{t['end']:.2f}] {spk_name}: {t['text']}"
        )

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print("\n✅ FINAL USER SPEAKER TRANSCRIPT CREATED")
    print(f"📄 Lines: {len(lines)}")
    print(f"📂 Output: {OUTPUT_FILE.resolve()}\n")


if __name__ == "__main__":
    main()
