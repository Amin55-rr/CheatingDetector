from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO


@dataclass
class DetectionResult:
    people_count: int
    phone_count: int
    offscreen: bool
    debug: Dict[str, float]


class ProctorDetector:
    def __init__(self, confidence_threshold: float = 0.45) -> None:
        self.model = YOLO("yolov8n.pt")
        self.confidence_threshold = confidence_threshold
        self.use_mediapipe_solutions = hasattr(mp, "solutions")
        self.face_mesh = None
        # Tête tournée (modéré) — pas trop serré pour limiter les faux positifs
        self.offscreen_threshold = 0.18
        # Yeux qui ne regardent plus l'écran (iris vs ouverture de l'œil)
        self.eye_away_threshold = 0.22
        self._yaw_history: List[float] = []
        self._yaw_window = 5
        if self.use_mediapipe_solutions:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self._fallback_baseline_x = None
            self._fallback_samples: List[float] = []
            self._fallback_warmup_samples = 45

    def _estimate_offscreen(self, frame_bgr: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        if not self.use_mediapipe_solutions:
            return self._estimate_offscreen_fallback(frame_bgr)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)  # type: ignore[union-attr]
        if not res.multi_face_landmarks:
            return False, {"face_detected": 0.0, "yaw_ratio": 0.0, "eye_offset": 0.0}

        face = res.multi_face_landmarks[0]
        h, w = frame_bgr.shape[:2]
        landmarks = face.landmark
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        nose_tip = landmarks[1]

        lx, rx, nx = left_eye.x * w, right_eye.x * w, nose_tip.x * w
        eye_mid_x = (lx + rx) / 2.0
        inter_eye = max(abs(rx - lx), 1.0)
        yaw_ratio = (nx - eye_mid_x) / inter_eye
        self._yaw_history.append(float(yaw_ratio))
        if len(self._yaw_history) > self._yaw_window:
            self._yaw_history.pop(0)
        smoothed_yaw = float(np.mean(self._yaw_history))

        eye_offscreen = False
        eye_offset = 0.0
        if len(landmarks) > 473:
            left_iris = landmarks[468]
            right_iris = landmarks[473]
            left_inner = landmarks[133]
            left_outer = landmarks[33]
            right_inner = landmarks[362]
            right_outer = landmarks[263]

            left_min_x = min(left_outer.x, left_inner.x) * w
            left_max_x = max(left_outer.x, left_inner.x) * w
            right_min_x = min(right_outer.x, right_inner.x) * w
            right_max_x = max(right_outer.x, right_inner.x) * w

            left_width = max(left_max_x - left_min_x, 1.0)
            right_width = max(right_max_x - right_min_x, 1.0)

            left_ratio = ((left_iris.x * w) - left_min_x) / left_width
            right_ratio = ((right_iris.x * w) - right_min_x) / right_width
            eye_offset = float(((left_ratio - 0.5) + (right_ratio - 0.5)) / 2.0)
            eye_offscreen = abs(eye_offset) > self.eye_away_threshold

        head_offscreen = abs(smoothed_yaw) > self.offscreen_threshold
        offscreen = head_offscreen or eye_offscreen
        return offscreen, {
            "face_detected": 1.0,
            "yaw_ratio": float(smoothed_yaw),
            "eye_offset": float(eye_offset),
            "head_offscreen": 1.0 if head_offscreen else 0.0,
            "eye_offscreen": 1.0 if eye_offscreen else 0.0,
        }

    def _estimate_offscreen_fallback(self, frame_bgr: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return False, {"face_detected": 0.0, "yaw_ratio": 0.0, "eye_offset": 0.0}

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        frame_w = frame_bgr.shape[1]
        face_center_x_norm = (x + (w / 2.0)) / max(float(frame_w), 1.0)

        if self._fallback_baseline_x is None:
            self._fallback_samples.append(face_center_x_norm)
            if len(self._fallback_samples) >= self._fallback_warmup_samples:
                self._fallback_baseline_x = float(np.mean(self._fallback_samples))
                self._fallback_samples.clear()
            return False, {"face_detected": 1.0, "yaw_ratio": 0.0, "eye_offset": 0.0}

        yaw_ratio = face_center_x_norm - self._fallback_baseline_x
        offscreen = abs(yaw_ratio) > 0.18
        return offscreen, {"face_detected": 1.0, "yaw_ratio": float(yaw_ratio), "eye_offset": 0.0}

    def analyze(self, frame_bgr: np.ndarray) -> DetectionResult:
        yolo_results = self.model.predict(
            source=frame_bgr,
            conf=self.confidence_threshold,
            verbose=False,
        )
        boxes = yolo_results[0].boxes
        people_count = 0
        phone_count = 0

        if boxes is not None:
            classes: List[int] = boxes.cls.cpu().numpy().astype(int).tolist()
            for cls_id in classes:
                if cls_id == 0:
                    people_count += 1
                elif cls_id == 67:
                    phone_count += 1

        offscreen, debug = self._estimate_offscreen(frame_bgr)
        return DetectionResult(
            people_count=people_count,
            phone_count=phone_count,
            offscreen=offscreen,
            debug=debug,
        )
