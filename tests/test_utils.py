"""Unit tests for src/utils.py functions."""

import numpy as np
import pytest

from src.utils import (
    LivenessTracker,
    check_challenge,
    encrypt_data,
    decrypt_data,
    estimate_head_yaw,
    eye_aspect_ratio,
    mouth_aspect_ratio,
)


class TestEyeAspectRatio:
    def test_open_eye_returns_high_ratio(self):
        eye = [
            (0, 0), (1, 2), (3, 2),
            (4, 0), (3, -2), (1, -2),
        ]
        ear = eye_aspect_ratio(eye)
        assert ear > 0.3

    def test_closed_eye_returns_low_ratio(self):
        eye = [
            (0, 0), (1, 0.1), (3, 0.1),
            (4, 0), (3, -0.1), (1, -0.1),
        ]
        ear = eye_aspect_ratio(eye)
        assert ear < 0.2

    def test_zero_width_returns_zero(self):
        eye = [
            (0, 0), (0, 1), (0, 1),
            (0, 0), (0, -1), (0, -1),
        ]
        ear = eye_aspect_ratio(eye)
        assert ear == 0.0


class TestMouthAspectRatio:
    def test_closed_mouth(self):
        mouth = [
            (0, 0), (1, 0), (2, 0.1), (3, 0),
            (4, 0), (3, 0), (2, -0.1), (1, 0),
        ]
        mar = mouth_aspect_ratio(mouth)
        assert mar < 0.3

    def test_open_mouth(self):
        mouth = [
            (0, 0), (1, 2), (2, 3), (3, 2),
            (4, 0), (3, -2), (2, -3), (1, -2),
        ]
        mar = mouth_aspect_ratio(mouth)
        assert mar > 0.5

    def test_too_few_points(self):
        assert mouth_aspect_ratio([(0, 0), (1, 1)]) == 0.0


class TestEstimateHeadYaw:
    def test_centered_face(self):
        landmarks = {
            "nose_bridge": [(50, 10), (50, 20), (50, 30), (50, 40)],
            "chin": [
                (10, i) for i in range(9)
            ] + [
                (90, i) for i in range(8)
            ],
        }
        landmarks["chin"] = [(10, 50)] + [(20, 50)] * 7 + [(50, 60)] * 4 + [(80, 50)] * 4 + [(90, 50)]
        yaw = estimate_head_yaw(landmarks)
        assert abs(yaw) < 20

    def test_missing_landmarks(self):
        assert estimate_head_yaw({}) == 0.0
        assert estimate_head_yaw({"nose_bridge": [(1, 1)]}) == 0.0


class TestEncryption:
    def test_roundtrip(self):
        data = b"hello biometric data"
        encrypted = encrypt_data(data)
        assert encrypted != data
        decrypted = decrypt_data(encrypted)
        assert decrypted == data

    def test_different_data_different_ciphertext(self):
        a = encrypt_data(b"data_a")
        b = encrypt_data(b"data_b")
        assert a != b


class TestLivenessTracker:
    def test_new_face_not_verified(self):
        tracker = LivenessTracker()
        assert not tracker.is_verified("face1")

    def test_blink_challenge_verification(self):
        tracker = LivenessTracker()
        state = tracker.get_state("face1")
        state["challenge"] = "BLINK"

        landmarks = {
            "left_eye": [(0, 0), (1, 0.05), (3, 0.05), (4, 0), (3, -0.05), (1, -0.05)],
            "right_eye": [(0, 0), (1, 0.05), (3, 0.05), (4, 0), (3, -0.05), (1, -0.05)],
        }
        for _ in range(3):
            tracker.update("face1", landmarks, 0.1, 2)

        open_landmarks = {
            "left_eye": [(0, 0), (1, 2), (3, 2), (4, 0), (3, -2), (1, -2)],
            "right_eye": [(0, 0), (1, 2), (3, 2), (4, 0), (3, -2), (1, -2)],
        }
        tracker.update("face1", open_landmarks, 0.4, 2)
        assert tracker.is_verified("face1")

    def test_reset_clears_state(self):
        tracker = LivenessTracker()
        state = tracker.get_state("face1")
        state["liveness_verified"] = True
        assert tracker.is_verified("face1")
        tracker.reset("face1")
        assert not tracker.is_verified("face1")

    def test_independent_faces(self):
        tracker = LivenessTracker()
        state1 = tracker.get_state("face1")
        state1["liveness_verified"] = True
        assert tracker.is_verified("face1")
        assert not tracker.is_verified("face2")
