"""ETL pipeline: sanitize images, extract 128-d embeddings, store in SQLite."""

import logging

import cv2
import face_recognition
import numpy as np

from src.config import ASSETS_DIR, DATA_DIR, setup_logging, validate_config
from src.database import init_db, save_encodings

logger = logging.getLogger(__name__)


def run_ingestion():
    setup_logging()
    validate_config()
    init_db()

    if not ASSETS_DIR.exists():
        ASSETS_DIR.mkdir(parents=True)
        logger.warning("No guest photos found. Created empty directory: %s", ASSETS_DIR)
        return

    DATA_DIR.mkdir(exist_ok=True)
    known_encodings = []
    known_names = []

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in ASSETS_DIR.iterdir() if p.suffix.lower() in valid_extensions]

    logger.info("Processing %d images...", len(images))

    for img_path in images:
        name = img_path.stem.replace("_", " ").title()

        img = cv2.imread(str(img_path))
        if img is None:
            logger.error("Could not read image: %s", name)
            continue

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            boxes = face_recognition.face_locations(gray, model="hog")
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb, boxes)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)
                logger.info("Indexed: %s", name)
            else:
                logger.warning("Skipped: %s (no face detected)", name)

        except Exception:
            logger.exception("Error processing %s", name)

    if known_names:
        save_encodings(known_names, known_encodings)
        logger.info("Database compiled. Total: %d", len(known_names))
    else:
        logger.warning("No faces found. Database is empty.")


if __name__ == "__main__":
    run_ingestion()
