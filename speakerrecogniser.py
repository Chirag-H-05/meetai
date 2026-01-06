import pickle
import torch
import numpy as np
from scipy.spatial.distance import cosine
from speechbrain.inference import EncoderClassifier

DB_PATH = "speaker_db.pkl"   # pre-enrolled speakers
THRESHOLD = 0.70
DEVICE = "cpu"   # keep stable

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE}
)

with open(DB_PATH, "rb") as f:
    SPEAKERS = pickle.load(f)


def identify_speaker(audio_np: np.ndarray):
    """
    audio_np: 1D numpy float32 @ 16kHz
    returns: (speaker_name, distance_score)
    """
    audio = torch.tensor(audio_np).unsqueeze(0)

    with torch.no_grad():
        emb = classifier.encode_batch(audio).squeeze().cpu()

    best_name = "Unknown"
    best_score = 1.0

    for name, ref_emb in SPEAKERS.items():
        score = cosine(emb, ref_emb)
        if score < best_score:
            best_score = score
            best_name = name

    if best_score > THRESHOLD:
        return "Unknown", best_score

    return best_name, best_score
