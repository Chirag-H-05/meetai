# download_models.py
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=  "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",
    cache_dir="models"
)

print("Models downloaded successfully.")
