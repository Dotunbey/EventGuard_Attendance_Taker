import cv2
import face_recognition
import csv
from datetime import datetime
from src.config import DB_PATH, LOG_PATH, EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, FRAME_RESIZE_SCALE, TOLERANCE
from src.utils import load_db, eye_aspect_ratio

class Gatekeeper:
    def __init__(self):
        self.db = load_db(DB_PATH)
        self.inside_guests = self._load_history()
        self.blink_consec_frames = 0
        self.liveness_verified = False
        self.current_user = None

    def _load_history(self) -> set:
        """Loads access log into memory set for O(1) lookup."""
        if not LOG_PATH.exists(): return set()
        with open(LOG_PATH, "r") as f:
            return {row[0] for row in csv.reader(f) if row}

    def _log_entry(self, name: str):
        self.inside_guests.add(name)
        file_exists = LOG_PATH.exists()
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(["Name", "Time", "Status"])
            writer.writerow([name, datetime.now().strftime("%H:%M:%S"), "ENTERED"])

    def process_frame(self, frame):
        # Resize for performance
        small = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        
        # Detection
        locs = face_recognition.face_locations(rgb_small)
        encs = face_recognition.face_encodings(rgb_small, locs)
        landmarks = face_recognition.face_landmarks(frame)

        # Liveness Check
        self._check_liveness(landmarks)

        # Identification
        for (top, right, bottom, left), enc in zip(locs, encs):
            self._draw_hud(frame, top*4, right*4, bottom*4, left*4, enc)

    def _check_liveness(self, landmarks_list):
        for lm in landmarks_list:
            left_ear = eye_aspect_ratio(lm['left_eye'])
            right_ear = eye_aspect_ratio(lm['right_eye'])
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EYE_AR_THRESH:
                self.blink_consec_frames += 1
            else:
                if self.blink_consec_frames >= EYE_AR_CONSEC_FRAMES:
                    self.liveness_verified = True
                self.blink_consec_frames = 0

    def _draw_hud(self, frame, t, r, b, l, encoding):
        matches = face_recognition.compare_faces(self.db["encodings"], encoding, tolerance=TOLERANCE)
        name, color, label = "Unknown", (0, 0, 255), "UNAUTHORIZED"

        if True in matches:
            idx = matches.index(True)
            name = self.db["names"][idx]
            
            if name in self.inside_guests:
                color, label = (0, 165, 255), "ALREADY INSIDE"
            elif self.liveness_verified:
                self._log_entry(name)
                color, label = (0, 255, 0), "ACCESS GRANTED"
                self.liveness_verified = False # Reset
            else:
                color, label = (0, 255, 255), "PLEASE BLINK"

        cv2.rectangle(frame, (l, t), (r, b), color, 2)
        cv2.putText(frame, f"{name}", (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(frame, label, (l, b+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            self.process_frame(frame)
            cv2.imshow("EventGuard Pro", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    Gatekeeper().run()
