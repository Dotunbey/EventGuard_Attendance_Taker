"""Gatekeeper Core: real-time face matching with per-face liveness verification."""

import logging
import os
import tempfile
from datetime import datetime

import cv2
import face_recognition
import numpy as np

from src.config import (
    EYE_AR_CONSEC_FRAMES,
    FRAME_RESIZE_SCALE,
    TOLERANCE,
    setup_logging,
    validate_config,
)
from src.database import (
    get_checked_in_names,
    init_db,
    is_inside,
    load_encodings,
    log_access,
)
from src.utils import LivenessTracker, eye_aspect_ratio

logger = logging.getLogger(__name__)


class Gatekeeper:
    def __init__(self):
        setup_logging()
        validate_config()
        init_db()

        try:
            self.db = load_encodings()
        except FileNotFoundError:
            logger.error("No encodings found. Run ingest.py first.")
            raise

        self.inside_guests = get_checked_in_names()
        self.liveness = LivenessTracker()
        logger.info(
            "Gatekeeper initialized: %d known faces, %d already inside",
            len(self.db["names"]),
            len(self.inside_guests),
        )

    def _face_id_from_encoding(self, encoding: np.ndarray) -> str:
        """Generate a stable face ID from the first few encoding values."""
        return str(hash(encoding[:8].tobytes()))

    def process_frame(self, frame):
        scale = FRAME_RESIZE_SCALE
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locs = face_recognition.face_locations(rgb_small)
        encs = face_recognition.face_encodings(rgb_small, locs)
        landmarks_list = face_recognition.face_landmarks(rgb_small)

        inv_scale = 1.0 / scale

        for i, ((top, right, bottom, left), enc) in enumerate(zip(locs, encs)):
            face_id = self._face_id_from_encoding(enc)

            if i < len(landmarks_list):
                lm = landmarks_list[i]
                left_ear = eye_aspect_ratio(lm["left_eye"])
                right_ear = eye_aspect_ratio(lm["right_eye"])
                avg_ear = (left_ear + right_ear) / 2.0
                self.liveness.update(face_id, lm, avg_ear, EYE_AR_CONSEC_FRAMES)

            orig_top = int(top * inv_scale)
            orig_right = int(right * inv_scale)
            orig_bottom = int(bottom * inv_scale)
            orig_left = int(left * inv_scale)

            self._draw_hud(
                frame, orig_top, orig_right, orig_bottom, orig_left, enc, face_id
            )

    def _draw_hud(self, frame, t, r, b, l, encoding, face_id):
        matches = face_recognition.compare_faces(
            self.db["encodings"], encoding, tolerance=TOLERANCE
        )
        name = "Unknown"
        color = (0, 0, 255)
        label = "UNAUTHORIZED"

        if True in matches:
            idx = matches.index(True)
            name = self.db["names"][idx]

            if name in self.inside_guests or is_inside(name):
                color, label = (0, 165, 255), "ALREADY INSIDE"
            elif self.liveness.is_verified(face_id):
                self._log_entry(name)
                color, label = (0, 255, 0), "ACCESS GRANTED"
                self.liveness.reset(face_id)
            else:
                challenge = self.liveness.get_challenge(face_id)
                color, label = (0, 255, 255), f"PLEASE: {challenge}"

        cv2.rectangle(frame, (l, t), (r, b), color, 2)
        cv2.putText(
            frame, name, (l, t - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2,
        )
        cv2.putText(
            frame, label, (l, b + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

    def _log_entry(self, name: str):
        """Record access event atomically via SQLite."""
        self.inside_guests.add(name)
        timestamp = datetime.now().strftime("%H:%M:%S")
        try:
            log_access(name, timestamp, "ENTERED")
            logger.info("ACCESS GRANTED: %s at %s", name, timestamp)
        except Exception:
            logger.exception("Failed to log access for %s", name)

    def run(self):
        cap = None
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.error("Cannot open camera (index 0)")
                return

            logger.info("Camera opened. Press 'q' to quit.")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break

                try:
                    self.process_frame(frame)
                except Exception:
                    logger.exception("Error processing frame")

                cv2.imshow("EventGuard Pro", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except Exception:
            logger.exception("Fatal error in gatekeeper run loop")
        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            logger.info("Gatekeeper shut down")


if __name__ == "__main__":
    Gatekeeper().run()
