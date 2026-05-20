"""Centralized configuration with validation and environment variable support."""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# -- Paths ----------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
ASSETS_DIR = BASE_DIR / "assets" / "guest_photos"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "encodings.pkl"  # legacy reference
SQLITE_DB_PATH = DATA_DIR / "eventguard.db"
LOG_PATH = DATA_DIR / "access_log.csv"  # legacy reference

# -- CV Constants ---------------------------------------------------------
FRAME_RESIZE_SCALE = float(os.getenv("EG_FRAME_RESIZE_SCALE", "0.5"))
TOLERANCE = float(os.getenv("EG_TOLERANCE", "0.5"))

# -- Multi-Person Tracking ------------------------------------------------
COOLDOWN_PERIOD = int(os.getenv("EG_COOLDOWN_PERIOD", "30"))
PROCESS_EVERY_N_FRAMES = int(os.getenv("EG_PROCESS_EVERY_N_FRAMES", "1"))
DETECTION_MODEL = os.getenv("EG_DETECTION_MODEL", "hog")
MIN_FACE_SIZE = int(os.getenv("EG_MIN_FACE_SIZE", "20"))
MAX_TRACKED_FACES = int(os.getenv("EG_MAX_TRACKED_FACES", "1000"))

# -- Encryption -----------------------------------------------------------
ENCRYPTION_KEY_ENV = "EVENTGUARD_ENCRYPTION_KEY"

# -- Logging --------------------------------------------------------------
LOG_LEVEL = os.getenv("EG_LOG_LEVEL", "INFO").upper()
LOG_FILE = DATA_DIR / "eventguard.log"

# -- Dashboard Auth -------------------------------------------------------
DASHBOARD_USERNAME = os.getenv("EG_DASHBOARD_USERNAME", "")
DASHBOARD_PASSWORD = os.getenv("EG_DASHBOARD_PASSWORD", "")


def validate_config():
    """Validate configuration values on startup. Exits on critical errors."""
    errors = []

    if not 0.0 < FRAME_RESIZE_SCALE <= 1.0:
        errors.append(
            f"FRAME_RESIZE_SCALE must be between 0 and 1, got {FRAME_RESIZE_SCALE}"
        )
    if not 0.0 < TOLERANCE <= 1.0:
        errors.append(
            f"TOLERANCE must be between 0 and 1, got {TOLERANCE}"
        )
    if COOLDOWN_PERIOD < 0:
        errors.append(
            f"COOLDOWN_PERIOD must be >= 0, got {COOLDOWN_PERIOD}"
        )
    if PROCESS_EVERY_N_FRAMES < 1:
        errors.append(
            f"PROCESS_EVERY_N_FRAMES must be >= 1, got {PROCESS_EVERY_N_FRAMES}"
        )
    if DETECTION_MODEL not in ("hog", "cnn"):
        errors.append(
            f"DETECTION_MODEL must be 'hog' or 'cnn', got {DETECTION_MODEL}"
        )
    if MIN_FACE_SIZE < 1:
        errors.append(
            f"MIN_FACE_SIZE must be >= 1, got {MIN_FACE_SIZE}"
        )
    if MAX_TRACKED_FACES < 1:
        errors.append(
            f"MAX_TRACKED_FACES must be >= 1, got {MAX_TRACKED_FACES}"
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
