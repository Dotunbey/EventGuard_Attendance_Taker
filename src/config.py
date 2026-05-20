"""Centralized configuration with validation and environment variable support."""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
ASSETS_DIR = BASE_DIR / "assets" / "guest_photos"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "encodings.pkl"  # legacy reference
SQLITE_DB_PATH = DATA_DIR / "eventguard.db"
LOG_PATH = DATA_DIR / "access_log.csv"  # legacy reference

# ── CV Constants ─────────────────────────────────────────────────
EYE_AR_THRESH = float(os.getenv("EG_EYE_AR_THRESH", "0.23"))
EYE_AR_CONSEC_FRAMES = int(os.getenv("EG_EYE_AR_CONSEC_FRAMES", "2"))
FRAME_RESIZE_SCALE = float(os.getenv("EG_FRAME_RESIZE_SCALE", "0.25"))
TOLERANCE = float(os.getenv("EG_TOLERANCE", "0.5"))

# ── Liveness Detection ──────────────────────────────────────────
MOUTH_AR_THRESH = float(os.getenv("EG_MOUTH_AR_THRESH", "0.6"))
HEAD_POSE_YAW_THRESH = float(os.getenv("EG_HEAD_POSE_YAW_THRESH", "15.0"))
CHALLENGE_TIMEOUT = int(os.getenv("EG_CHALLENGE_TIMEOUT", "10"))  # seconds

# ── Encryption ───────────────────────────────────────────────────
ENCRYPTION_KEY_ENV = "EVENTGUARD_ENCRYPTION_KEY"

# ── Logging ──────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("EG_LOG_LEVEL", "INFO").upper()
LOG_FILE = DATA_DIR / "eventguard.log"

# ── Dashboard Auth ───────────────────────────────────────────────
DASHBOARD_USERNAME = os.getenv("EG_DASHBOARD_USERNAME", "")
DASHBOARD_PASSWORD = os.getenv("EG_DASHBOARD_PASSWORD", "")


def validate_config():
    """Validate configuration values on startup. Exits on critical errors."""
    errors = []

    if not 0.0 < EYE_AR_THRESH < 1.0:
        errors.append(
            f"EYE_AR_THRESH must be between 0 and 1, got {EYE_AR_THRESH}"
        )
    if EYE_AR_CONSEC_FRAMES < 1:
        errors.append(
            f"EYE_AR_CONSEC_FRAMES must be >= 1, got {EYE_AR_CONSEC_FRAMES}"
        )
    if not 0.0 < FRAME_RESIZE_SCALE <= 1.0:
        errors.append(
            f"FRAME_RESIZE_SCALE must be between 0 and 1, got {FRAME_RESIZE_SCALE}"
        )
    if not 0.0 < TOLERANCE <= 1.0:
        errors.append(
            f"TOLERANCE must be between 0 and 1, got {TOLERANCE}"
        )
    if not 0.0 < MOUTH_AR_THRESH < 2.0:
        errors.append(
            f"MOUTH_AR_THRESH must be between 0 and 2, got {MOUTH_AR_THRESH}"
        )
    if HEAD_POSE_YAW_THRESH <= 0:
        errors.append(
            f"HEAD_POSE_YAW_THRESH must be > 0, got {HEAD_POSE_YAW_THRESH}"
        )
    if CHALLENGE_TIMEOUT < 1:
        errors.append(
            f"CHALLENGE_TIMEOUT must be >= 1, got {CHALLENGE_TIMEOUT}"
        )

    if errors:
        for e in errors:
            logger.error("Config validation failed: %s", e)
        sys.exit(1)

    logger.debug("Configuration validated successfully")


def setup_logging():
    """Configure application-wide logging."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    # File handler
    try:
        file_handler = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except OSError as exc:
        root_logger.warning("Could not set up file logging: %s", exc)
