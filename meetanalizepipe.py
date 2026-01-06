import subprocess
from pathlib import Path
import sys

# ============================================================
# CONFIG — CHANGE ONLY IF NEEDED
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"
INPUT_FILE = OUTPUTS_DIR / "speaker_transcript.txt"

# ⚠️ REQUIRED on Windows: absolute path to ollama.exe
OLLAMA_EXE = Path(
    r"C:/Users/ISFL-RT000268/AppData/Local/Programs/Ollama/ollama.exe"
)

# ============================================================
# SAFETY CHECKS
# ============================================================
def fail(msg):
    print(f"\n❌ ERROR: {msg}\n")
    sys.exit(1)

if not INPUT_FILE.exists():
    fail(f"Missing input file: {INPUT_FILE.resolve()}")

if not OLLAMA_EXE.exists():
    fail(f"Ollama not found at: {OLLAMA_EXE}")

REPORTS_DIR.mkdir(exist_ok=True)

# ============================================================
# OLLAMA RUNNER
# ============================================================
def run_ollama(model: str, prompt: str, input_text: str, output_file: Path):
    """
    Runs ollama with a prompt + input text.
    Writes stdout to output_file.
    """
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
        raise RuntimeError(
            f"Ollama failed ({model}):\n{result.stderr}"
        )

    output_file.write_text(result.stdout.strip(), encoding="utf-8")

# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    transcript = INPUT_FILE.read_text(encoding="utf-8")

    # --------------------------------------------------------
    # STEP 1 — STRUCTURE & NORMALIZE TRANSCRIPT
    # --------------------------------------------------------
    print("▶ Step 1: Structuring & normalizing transcript")

    structure_prompt = """
You are a meeting transcript normalizer and structurer.

GOAL:
Create a clean, speaker-attributed, chronologically accurate record.

RULES:
- Do NOT summarize
- Do NOT infer intent
- Do NOT merge speakers
- Do NOT remove disagreements
- Preserve technical, business, and timeline details

CLEANING:
- Remove filler words, stutters, false starts
- Keep exact meaning
- Preserve numbers, dates, names, commitments

OUTPUT FORMAT (STRICT):

MEETING_METADATA:
- Date (if mentioned)
- Time (if mentioned)
- Context / Purpose (from transcript only)

SPEAKER_STATEMENTS:
SPEAKER_1:
- [timestamp if present] Exact cleaned statement
- [timestamp if present] Exact cleaned statement

SPEAKER_2:
- ...

UNRESOLVED_REFERENCES:
- Items mentioned but not concluded
"""

    structured_file = REPORTS_DIR / "01_structured_transcript.txt"
    run_ollama(
        model="vicuna:13b",
        prompt=structure_prompt,
        input_text=transcript,
        output_file=structured_file
    )

    structured = structured_file.read_text(encoding="utf-8")

    # --------------------------------------------------------
    # STEP 2 — FULL IN-DEPTH MEETING REPORT
    # --------------------------------------------------------
    print("▶ Step 2: Generating full in-depth meeting report")

    analysis_prompt = """
You are a senior program analyst producing a COMPLETE, IN-DEPTH MEETING REPORT.

THIS IS NOT A SUMMARY.
THIS IS NOT HIGH LEVEL.
THIS MUST ALLOW SOMEONE TO FULLY UNDERSTAND THE MEETING WITHOUT ATTENDING IT.

ANALYZE THE CONTENT AND PRODUCE THE FOLLOWING SECTIONS EXACTLY:

--------------------------------------------------
1. MEETING PURPOSE & CONTEXT
--------------------------------------------------
- Why this meeting occurred
- Triggering events or problems
- Background referenced by speakers

--------------------------------------------------
2. PHASE-WISE DISCUSSION BREAKDOWN
--------------------------------------------------
For EACH phase or topic discussed:

PHASE NAME:
- Start indicator (what caused this phase)
- Core discussion points
- Technical / business details
- Risks mentioned
- Constraints mentioned
- Conflicts or disagreements

--------------------------------------------------
3. DECISIONS MADE
--------------------------------------------------
For EACH decision:
- Decision description
- Exact wording source (speaker + paraphrase)
- Decision type:
  [Approved / Rejected / Deferred / Tentative]
- Conditions (if any)
- Confidence level (explicit or implied)

--------------------------------------------------
4. ACTION ITEMS & RESPONSIBILITY
--------------------------------------------------
For EACH action:
- Action description
- Responsible person(s)
- Supporting speakers
- Deadline or timeframe (explicit or implied)
- Dependency (if any)

--------------------------------------------------
5. RECOMMENDATIONS RAISED
--------------------------------------------------
For EACH recommendation:
- Who proposed it
- Rationale
- Acceptance status
- Pushback (if any)

--------------------------------------------------
6. SPEAKER-BY-SPEAKER PERSPECTIVE ANALYSIS
--------------------------------------------------
For EACH speaker:

SPEAKER_X:
- Role (inferred from speech ONLY)
- Primary objectives
- Key concerns
- Arguments made
- Stance on decisions
- Conflicts with others
- Influence level (Low / Medium / High)

--------------------------------------------------
7. CURRENT STATE (END OF MEETING)
--------------------------------------------------
- What is finalized
- What is pending
- What remains unclear
- Risks carried forward

--------------------------------------------------
8. NEXT PHASE & SCHEDULE
--------------------------------------------------
- Next steps agreed
- Timeline references (dates / weeks / milestones)
- Ownership mapping
- Open dependencies

--------------------------------------------------
9. OPEN QUESTIONS & UNRESOLVED ISSUES
--------------------------------------------------
- Exact unresolved points
- Who raised them
- Why unresolved

RULES:
- Cite speakers everywhere
- Do NOT invent facts
- Do NOT generalize
- Be exhaustive
"""

    run_ollama(
        model="deepseek-r1:14b",
        prompt=analysis_prompt,
        input_text=structured,
        output_file=REPORTS_DIR / "02_full_meeting_report.md"
    )

    # --------------------------------------------------------
    # STEP 3 — EXECUTIVE SUMMARY (DERIVED)
    # --------------------------------------------------------
    print("▶ Step 3: Creating executive summary")

    summary_prompt = """
Create an executive summary BASED STRICTLY on the detailed report.

FORMAT:
- Bullet points only
- No speculation
- No new information

INCLUDE:
- Core objective
- Major decisions
- High-risk items
- Immediate next steps
- Ownership clarity
"""

    run_ollama(
        model="llama3",
        prompt=summary_prompt,
        input_text=(REPORTS_DIR / "02_full_meeting_report.md").read_text(encoding="utf-8"),
        output_file=REPORTS_DIR / "03_executive_summary.md"
    )

    # --------------------------------------------------------
    # STEP 4 — SPEAKER INTELLIGENCE REPORT
    # --------------------------------------------------------
    print("▶ Step 4: Speaker intelligence analysis")

    speaker_prompt = """
Produce a deep speaker intelligence report.

FOR EACH SPEAKER:

- Inferred role
- Decision alignment
- Risk appetite
- Communication style
- Areas of resistance
- Areas of influence
- Trustworthiness of commitments (based on consistency)

RULES:
- Base everything ONLY on transcript evidence
- No psychology beyond observable behavior
"""

    run_ollama(
        model="deepseek-r1:14b",
        prompt=speaker_prompt,
        input_text=structured,
        output_file=REPORTS_DIR / "04_speaker_intelligence.md"
    )

    print("\n✅ MEETING ANALYSIS COMPLETE")
    print(f"📂 Reports saved to: {REPORTS_DIR.resolve()}\n")

# ============================================================
# ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    main()
