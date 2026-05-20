"""Shared CV utilities: EAR, liveness helpers, and encrypted storage."""

import base64
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import distance as dist

from src.config import (
    ENCRYPTION_KEY_ENV,
    EYE_AR_THRESH,
    HEAD_POSE_YAW_THRESH,
    MOUTH_AR_THRESH,
)

logger = logging.getLogger(__name__)


# ── Eye Aspect Ratio ─────────────────────────────────────────────

def eye_aspect_ratio(eye) -> float:
    """Calculate Eye Aspect Ratio for blink detection."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


# ── Mouth Aspect Ratio ──────────────────────────────────────────

def mouth_aspect_ratio(mouth_points) -> float:
    """Calculate Mouth Aspect Ratio for mouth-open detection.

    Uses the inner lip landmarks (top_lip / bottom_lip from face_recognition).
    """
    if len(mouth_points) < 8:
        return 0.0
    A = dist.euclidean(mouth_points[2], mouth_points[6])
    B = dist.euclidean(mouth_points[3], mouth_points[5])
    C = dist.euclidean(mouth_points[0], mouth_points[4])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


# ── Head Pose Estimation ────────────────────────────────────────

def estimate_head_yaw(landmarks: dict) -> float:
    """Estimate horizontal head rotation from landmark positions.

    Uses the nose bridge and face width to approximate yaw angle.
    Positive = looking right, negative = looking left.
    """
    nose_bridge = landmarks.get("nose_bridge", [])
    chin = landmarks.get("chin", [])

    if len(nose_bridge) < 2 or len(chin) < 9:
        return 0.0

    nose_tip = np.array(nose_bridge[-1], dtype=np.float64)
    left_jaw = np.array(chin[0], dtype=np.float64)
    right_jaw = np.array(chin[-1], dtype=np.float64)

    face_center_x = (left_jaw[0] + right_jaw[0]) / 2.0
    face_width = abs(right_jaw[0] - left_jaw[0])

    if face_width == 0:
        return 0.0

    offset = (nose_tip[0] - face_center_x) / face_width
    yaw_degrees = offset * 90.0
    return float(yaw_degrees)


# ── Challenge-Response ───────────────────────────────────────────

CHALLENGES = ["LOOK LEFT", "LOOK RIGHT", "OPEN MOUTH", "BLINK"]


def generate_challenge() -> str:
    """Generate a random liveness challenge prompt."""
    return random.choice(CHALLENGES)


def check_challenge(
    challenge: str,
    landmarks: dict,
    avg_ear: float,
    blink_detected: bool,
) -> bool:
    """Check whether the current frame satisfies the active challenge."""
    if challenge == "BLINK":
        return blink_detected
    elif challenge == "LOOK LEFT":
        yaw = estimate_head_yaw(landmarks)
        return yaw < -HEAD_POSE_YAW_THRESH
    elif challenge == "LOOK RIGHT":
        yaw = estimate_head_yaw(landmarks)
        return yaw > HEAD_POSE_YAW_THRESH
    elif challenge == "OPEN MOUTH":
        bottom_lip = landmarks.get("bottom_lip", [])
        top_lip = landmarks.get("top_lip", [])
        if top_lip and bottom_lip:
            inner_mouth = top_lip + bottom_lip
            mar = mouth_aspect_ratio(inner_mouth)
            return mar > MOUTH_AR_THRESH
    return False


# ── Per-Face Liveness Tracker ────────────────────────────────────

class LivenessTracker:
    """Track blink / challenge state per detected face."""

    def __init__(self):
        self._states: Dict[str, dict] = {}

    def get_state(self, face_id: str) -> dict:
        if face_id not in self._states:
            self._states[face_id] = {
                "blink_frames": 0,
                "liveness_verified": False,
                "challenge": generate_challenge(),
                "challenge_start": time.time(),
                "mouth_open_frames": 0,
            }
        return self._states[face_id]

    def update(
        self,
        face_id: str,
        landmarks: dict,
        avg_ear: float,
        consec_thresh: int,
    ):
        """Update liveness state for a single face."""
        state = self.get_state(face_id)

        if state["liveness_verified"]:
            return

        blink_detected = False
        if avg_ear < EYE_AR_THRESH:
            state["blink_frames"] += 1
        else:
            if state["blink_frames"] >= consec_thresh:
                blink_detected = True
            state["blink_frames"] = 0

        if check_challenge(state["challenge"], landmarks, avg_ear, blink_detected):
            state["liveness_verified"] = True
            logger.info("Liveness verified for face %s (challenge: %s)", face_id, state["challenge"])

    def is_verified(self, face_id: str) -> bool:
        return self.get_state(face_id).get("liveness_verified", False)

    def get_challenge(self, face_id: str) -> str:
        return self.get_state(face_id).get("challenge", "BLINK")

    def reset(self, face_id: str):
        if face_id in self._states:
            del self._states[face_id]

    def clear(self):
        self._states.clear()


# ── Encryption helpers ───────────────────────────────────────────

def _get_encryption_key() -> bytes:
    """Derive a 32-byte Fernet-compatible key from the environment variable."""
    try:
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives import hashes
    except ImportError:
        raise ImportError(
            "cryptography package is required for encrypted storage. "
            "Install it with: pip install cryptography"
        )

    key_material = os.getenv(ENCRYPTION_KEY_ENV, "")
    if not key_material:
        logger.warning(
            "No encryption key found in %s. Using default key. "
            "Set %s for production use.",
            ENCRYPTION_KEY_ENV, ENCRYPTION_KEY_ENV,
        )
        key_material = "eventguard-default-dev-key"

    salt = b"eventguard-salt-v1"
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000,
    )
    derived = kdf.derive(key_material.encode("utf-8"))
    return base64.urlsafe_b64encode(derived)


def encrypt_data(data: bytes) -> bytes:
    """Encrypt data using Fernet (AES-128-CBC under the hood)."""
    from cryptography.fernet import Fernet

    key = _get_encryption_key()
    f = Fernet(key)
    return f.encrypt(data)


def decrypt_data(token: bytes) -> bytes:
    """Decrypt Fernet-encrypted data."""
    from cryptography.fernet import Fernet

    key = _get_encryption_key()
    f = Fernet(key)
    return f.decrypt(token)


# ── Legacy load/save kept for migration compatibility ────────────

def load_db(path) -> Dict[str, Any]:
    """Load legacy pickle database (kept for backward compat)."""
    logger.warning("Using legacy pickle load from %s", path)
    import pickle

    if not path.exists():
        raise FileNotFoundError(
            f"Database not found at {path}. Run ingest.py first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def save_db(data: Dict[str, Any], path):
    """Save legacy pickle database (kept for backward compat)."""
    logger.warning("Using legacy pickle save to %s", path)
    import pickle

    with open(path, "wb") as f:
        pickle.dump(data, f)
