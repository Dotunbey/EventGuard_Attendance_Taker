from scipy.spatial import distance as dist
import pickle
from typing import Dict, Any

def eye_aspect_ratio(eye) -> float:
    """Calculates EAR for liveness detection."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def load_db(path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Database not found at {path}. Run ingest.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)

def save_db(data: Dict[str, Any], path):
    with open(path, "wb") as f:
        pickle.dump(data, f)
