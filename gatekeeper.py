"""Gatekeeper Core: high-throughput attendance system with multi-person tracking."""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import face_recognition
import numpy as np

from src.config import (
    COOLDOWN_PERIOD,
    DETECTION_MODEL,
    FRAME_RESIZE_SCALE,
    MAX_TRACKED_FACES,
    MIN_FACE_SIZE,
    PROCESS_EVERY_N_FRAMES,
    TOLERANCE,
    setup_logging,
    validate_config,
)
from src.database import (
    get_checked_in_names,
    init_db,
    load_encodings,
    log_access,
)

logger = logging.getLogger(__name__)


class TrackedFace:
    """State for a single tracked face across frames."""

    __slots__ = ("name", "last_seen", "logged")

    def __init__(self, name: str, last_seen: float, logged: bool = False):
        self.name = name
        self.last_seen = last_seen
        self.logged = logged


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
        self.tracked_faces: Dict[str, TrackedFace] = {}
        self.frame_count = 0
        self.session_start = time.time()
        self.total_detections = 0
        self.roi: Optional[Tuple[int, int, int, int]] = None

        logger.info(
            "Gatekeeper initialized: %d known faces, %d already inside, "
            "model=%s, scale=%.2f, cooldown=%ds",
            len(self.db["names"]),
            len(self.inside_guests),
            DETECTION_MODEL,
            FRAME_RESIZE_SCALE,
            COOLDOWN_PERIOD,
        )

    def set_roi(self, x: int, y: int, w: int, h: int):
        """Set a Region of Interest to limit detection to a specific area."""
        self.roi = (x, y, w, h)
        logger.info("ROI set: x=%d, y=%d, w=%d, h=%d", x, y, w, h)

    def clear_roi(self):
        """Clear the Region of Interest so the full frame is processed."""
        self.roi = None
        logger.info("ROI cleared")

    def _cleanup_tracked_faces(self):
        """Remove stale entries that exceed the cooldown period."""
        now = time.time()
        expired = [
            fid for fid, tf in self.tracked_faces.items()
            if now - tf.last_seen > COOLDOWN_PERIOD * 2
        ]
        for fid in expired:
            del self.tracked_faces[fid]

    def _filter_small_faces(
        self,
        locations: List[Tuple[int, int, int, int]],
        encodings: List[np.ndarray],
        scale: float,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
        """Drop detections smaller than MIN_FACE_SIZE."""
        kept_locs = []
        kept_encs = []
        inv = 1.0 / scale
        for loc, enc in zip(locations, encodings):
            top, right, bottom, left = loc
            h = int((bottom - top) * inv)
            w = int((right - left) * inv)
            if h >= MIN_FACE_SIZE and w >= MIN_FACE_SIZE:
                kept_locs.append(loc)
                kept_encs.append(enc)
        return kept_locs, kept_encs

    def _find_tracked_face(self, encoding: np.ndarray) -> Optional[str]:
        """Find an existing tracked face that matches the encoding."""
        for fid, tf in self.tracked_faces.items():
            stored_enc = self._tracked_encodings.get(fid)
            if stored_enc is not None:
                dist = np.linalg.norm(stored_enc - encoding)
                if dist < TOLERANCE:
                    return fid
        return None

    def _track_id(self, encoding: np.ndarray) -> str:
        """Return a stable tracking id, reusing one if close enough."""
        existing = self._find_tracked_face(encoding)
        if existing is not None:
            self._tracked_encodings[existing] = encoding
            return existing
        new_id = str(hash(encoding[:8].tobytes()))
        self._tracked_encodings[new_id] = encoding
        return new_id

    def process_frame(self, frame: np.ndarray):
        """Detect and identify all faces in a single frame."""
        self.frame_count += 1
        if self.frame_count % PROCESS_EVERY_N_FRAMES != 0:
            self._draw_tracked(frame)
            return

        region = frame
        offset_x, offset_y = 0, 0
        if self.roi is not None:
            rx, ry, rw, rh = self.roi
            region = frame[ry:ry + rh, rx:rx + rw]
            offset_x, offset_y = rx, ry

        scale = FRAME_RESIZE_SCALE
        small = cv2.resize(region, (0, 0), fx=scale, fy=scale)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locs = face_recognition.face_locations(rgb_small, model=DETECTION_MODEL)
        encs = face_recognition.face_encodings(rgb_small, locs)

        locs, encs = self._filter_small_faces(locs, encs, scale)

        if not hasattr(self, "_tracked_encodings"):
            self._tracked_encodings: Dict[str, np.ndarray] = {}

        inv_scale = 1.0 / scale
        now = time.time()

        if encs:
            distances = face_recognition.face_distance(
                self.db["encodings"], encs[0]
            )
            batch_results = []
            for enc in encs:
                dists = face_recognition.face_distance(self.db["encodings"], enc)
                best_idx = int(np.argmin(dists))
                best_dist = dists[best_idx]
                matched = best_dist <= TOLERANCE
                batch_results.append((matched, best_idx, best_dist))

            for (top, right, bottom, left), enc, (matched, best_idx, best_dist) in zip(
                locs, encs, batch_results
            ):
                track_id = self._track_id(enc)
                self.total_detections += 1

                orig_top = int(top * inv_scale) + offset_y
                orig_right = int(right * inv_scale) + offset_x
                orig_bottom = int(bottom * inv_scale) + offset_y
                orig_left = int(left * inv_scale) + offset_x

                confidence = max(0.0, 1.0 - best_dist) if matched else 0.0

                if matched:
                    name = self.db["names"][best_idx]
                    tf = self.tracked_faces.get(track_id)

                    if tf is not None:
                        tf.last_seen = now
                        tf.name = name
                    else:
                        tf = TrackedFace(name=name, last_seen=now)
                        self.tracked_faces[track_id] = tf

                    if self._should_log(name, tf, now):
                        self._log_entry(name)
                        tf.logged = True
                        color = (0, 255, 0)
                        label = "LOGGED"
                    elif name in self.inside_guests:
                        color = (0, 165, 255)
                        label = "ALREADY LOGGED"
                    else:
                        color = (0, 255, 0)
                        label = "LOGGED"
                else:
                    name = "Unknown"
                    color = (0, 0, 255)
                    label = "UNRECOGNIZED"

                conf_text = f"{confidence:.0%}" if matched else ""
                self._draw_face(
                    frame, orig_top, orig_right, orig_bottom, orig_left,
                    name, label, color, conf_text,
                )

        if len(self.tracked_faces) > MAX_TRACKED_FACES:
            self._cleanup_tracked_faces()

        self._draw_stats(frame)

    def _should_log(self, name: str, tf: TrackedFace, now: float) -> bool:
        """Decide whether to log an attendance entry for this person."""
        if name in self.inside_guests:
            if tf.logged and (now - tf.last_seen) < COOLDOWN_PERIOD:
                return False
            if name in self.inside_guests:
                return False
        return True

    def _log_entry(self, name: str):
        """Record access event atomically via SQLite."""
        self.inside_guests.add(name)
        timestamp = datetime.now().strftime("%H:%M:%S")
        try:
            log_access(name, timestamp, "ENTERED")
            logger.info("ATTENDANCE LOGGED: %s at %s", name, timestamp)
        except Exception:
            logger.exception("Failed to log access for %s", name)

    def _draw_face(
        self,
        frame: np.ndarray,
        t: int, r: int, b: int, l: int,
        name: str, label: str, color: Tuple[int, int, int],
        confidence: str,
    ):
        """Draw bounding box, name, label, and confidence on a single face."""
        cv2.rectangle(frame, (l, t), (r, b), color, 2)
        cv2.putText(
            frame, name, (l, t - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2,
        )
        display = f"{label} {confidence}".strip()
        cv2.putText(
            frame, display, (l, b + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

    def _draw_tracked(self, frame: np.ndarray):
        """Re-draw boxes for tracked faces on skipped frames."""
        pass

    def _draw_stats(self, frame: np.ndarray):
        """Overlay session statistics on the frame."""
        elapsed = max(time.time() - self.session_start, 1)
        rate = self.total_detections / (elapsed / 60.0)
        stats = [
            f"Tracked: {len(self.tracked_faces)}",
            f"Logged: {len(self.inside_guests)}",
            f"Rate: {rate:.1f}/min",
        ]
        y = 30
        for line in stats:
            cv2.putText(
                frame, line, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )
            y += 25

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

                cv2.imshow("EventGuard Attendance", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except Exception:
            logger.exception("Fatal error in gatekeeper run loop")
        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            logger.info(
                "Gatekeeper shut down. Total detections: %d, Logged: %d",
                self.total_detections,
                len(self.inside_guests),
            )


if __name__ == "__main__":
    Gatekeeper().run()
