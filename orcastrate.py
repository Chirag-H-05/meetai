import sys
import os
import re
import subprocess
from pathlib import Path

import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# ============================================================
# CLI INPUT
# ============================================================
if len(sys.argv) != 2:
    print("\n❌ Usage: python meeting_pipeline.py <audio_file_path>\n")
    sys.exit(1)

AUDIO_FILE = sys.argv[1]

if not os.path.exists(AUDIO_FILE):
    print(f"\n❌ Audio file not found: {AUDIO_FILE}\n")
    sys.exit(1)

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"

OUTPUTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

TRANSCRIPT_FILE = OUTPUTS_DIR / "transcript.txt"
RTTM_FILE = OUTPUTS_DIR / "diarization.rttm"
DIAR_TXT_FILE = OUTPUTS_DIR / "diarization.txt"
SPEAKER_TXT_FILE = OUTPUTS_DIR / "speaker_transcript.txt"

# ⚠️ Windows Ollama path (UNCHANGED)
OLLAMA_EXE = Path(
    r"C:/Users/ISFL-RT000268/AppData/Local/Programs/Ollama/ollama.exe"
)

if not OLLAMA_EXE.exists():
    print(f"\n❌ Ollama not found at {OLLAMA_EXE}\n")
    sys.exit(1)

# ============================================================
# STEP 1 — TRANSCRIPTION (FASTER-WHISPER)
# ============================================================
print("\n▶ STEP 1: Transcription started")

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

with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
    for seg in segments:
        line = f"[{seg.start:7.2f} → {seg.end:7.2f}] {seg.text.strip()}"
        f.write(line + "\n")

print("✅ Transcription completed")
print("Detected language:", info.language)
print("Saved:", TRANSCRIPT_FILE.resolve())

# ============================================================
# STEP 2 — SPEAKER DIARIZATION (PYANNOTE)
# ============================================================
print("\n▶ STEP 2: Speaker diarization started")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    cache_dir="models"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

diarization = pipeline(AUDIO_FILE)

with open(RTTM_FILE, "w") as f:
    diarization.write_rttm(f)

with open(DIAR_TXT_FILE, "w") as f:
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        line = f"{segment.start:.2f} {segment.end:.2f} {speaker}"
        f.write(line + "\n")

print("✅ Diarization completed")
print("Saved:", RTTM_FILE.resolve())

# ============================================================
# STEP 3 — SPEAKER ↔ TRANSCRIPT ALIGNMENT
# ============================================================
print("\n▶ STEP 3: Creating speaker transcript")

def load_rttm(path: Path):
    segments = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            start = float(parts[3])
            dur = float(parts[4])
            speaker = parts[7]
            segments.append({
                "start": start,
                "end": start + dur,
                "speaker": speaker
            })
    return segments


def load_transcript(path: Path):
    pattern = re.compile(r"\[(\d+\.\d+)\s*→\s*(\d+\.\d+)\]\s*(.*)")
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
    return chunks


def find_speaker(start, diarization):
    for d in diarization:
        if d["start"] <= start <= d["end"]:
            return d["speaker"]
    return "UNKNOWN"


diar = load_rttm(RTTM_FILE)
trans = load_transcript(TRANSCRIPT_FILE)

lines = []
for t in trans:
    speaker = find_speaker(t["start"], diar)
    lines.append(
        f"[{t['start']:.2f}–{t['end']:.2f}] {speaker}: {t['text']}"
    )

with open(SPEAKER_TXT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("✅ Speaker transcript created")
print("Saved:", SPEAKER_TXT_FILE.resolve())

# ============================================================
# STEP 4 — MEETING ANALYSIS (OLLAMA)
# ============================================================
print("\n▶ STEP 4: Meeting analysis started")

def run_ollama(model: str, prompt: str, input_text: str, output_file: Path):
    full_prompt = prompt.strip() + "\n\n" + input_text.strip()
    result = subprocess.run(
        [str(OLLAMA_EXE), "run", model],
        input=full_prompt,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace"
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    output_file.write_text(result.stdout.strip(), encoding="utf-8")

speaker_transcript = SPEAKER_TXT_FILE.read_text(encoding="utf-8")

# ---- Structuring
run_ollama(
    "vicuna:13b",
    """
You are a meeting transcription cleaner.
TASK:
- Remove transcription noise and repetition
- Group content by speaker
- Preserve factual meaning
- DO NOT summarize or add interpretation
FORMAT:
SPEAKER_1:
- bullet points
SPEAKER_2:
- bullet points
""",
    speaker_transcript,
    REPORTS_DIR / "structured.txt"
)

structured = (REPORTS_DIR / "structured.txt").read_text(encoding="utf-8")

# ---- Outcomes
run_ollama(
    "deepseek-r1:14b",
    """
Analyze this meeting and extract:
- Key discussion topics
- Decisions made
- Commitments / promises
- Numbers, budgets, timelines
- Conflicts or disagreements
- Unresolved issues
RULES:
- Be precise
- Cite speakers
- No speculation
""",
    structured,
    REPORTS_DIR / "outcomes.md"
)

# ---- Summary
run_ollama(
    "llama3",
    """
Write an executive summary.
Audience: Senior leadership
Tone: Neutral, factual
Format: Bullet points
Focus on:
- What happened
- Why it matters
- What happens next
""",
    structured,
    REPORTS_DIR / "summary.md"
)

# ---- Speaker Analysis
run_ollama(
    "deepseek-r1:14b",
    """
For each speaker, provide:
- Inferred role
- Main arguments
- Tone (e.g. defensive, assertive, critical)
- Influence on decisions
Base everything strictly on the transcript.
""",
    structured,
    REPORTS_DIR / "speaker_analysis.md"
)

print("\n✅ MEETING PIPELINE COMPLETE")
print("📂 Outputs:", OUTPUTS_DIR.resolve())
print("📂 Reports:", REPORTS_DIR.resolve())
