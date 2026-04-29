import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from app.config import SessionConfig
from app.detectors import ProctorDetector
from app.logger import AlertLogger


class ProctorSession:
    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self.session_dir: Optional[Path] = None
        self.logger: Optional[AlertLogger] = None
        self.detector = ProctorDetector(confidence_threshold=config.confidence_threshold)
        self.offscreen_start_ts: Optional[float] = None
        self.offscreen_last_seen_ts: Optional[float] = None
        self.multiple_people_start_ts: Optional[float] = None
        self.alert_count = 0
        self.start_ts = time.time()
        self.last_alert_ts_by_type: Dict[str, float] = {}

    def _bootstrap_session_workspace(self) -> None:
        if self.logger is not None and self.session_dir is not None:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path("sessions") / f"{self.config.student_id}_{stamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir = session_dir
        self.logger = AlertLogger(session_dir)

    def _save_capture(self, frame, event_type: str) -> Optional[str]:
        if not self.config.capture_on_alert:
            return None
        if self.logger is None:
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{event_type}.jpg"
        path = self.logger.captures_dir / filename
        cv2.imwrite(str(path), frame)
        return str(path)

    def _emit_alert(self, frame, event_type: str, severity: str, meta: Dict, now: float) -> bool:
        if self.logger is None:
            return False
        last_ts = self.last_alert_ts_by_type.get(event_type, 0.0)
        if now - last_ts < self.config.alert_cooldown_seconds:
            return False

        capture = self._save_capture(frame, event_type)
        if capture:
            meta = {**meta, "capture": capture}
        self.logger.add_event(event_type=event_type, severity=severity, meta=meta)
        self.alert_count += 1
        self.last_alert_ts_by_type[event_type] = now
        return True

    def run(self) -> None:
        cap = cv2.VideoCapture(self.config.webcam_index)
        if not cap.isOpened():
            raise RuntimeError("Webcam inaccessible. Vérifiez la caméra et les permissions.")

        self._bootstrap_session_workspace()

        duration_seconds = self.config.duration_minutes * 60
        print(f"Session démarrée: {self.config.student_id} | Matière: {self.config.subject}")
        print("Appuyez sur 'q' pour arrêter manuellement.")

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            elapsed = time.time() - self.start_ts
            if elapsed >= duration_seconds:
                print("Fin automatique: durée atteinte.")
                break

            result = self.detector.analyze(frame)
            now = time.time()
            active_alerts: List[str] = []

            if result.phone_count > 0:
                emitted = self._emit_alert(
                    frame,
                    "phone_detected",
                    "critical",
                    {
                        "phone_count": result.phone_count,
                        "rule": "telephone_visible",
                    },
                    now,
                )
                if emitted:
                    active_alerts.append("phone_detected")

            if result.people_count >= self.config.multiple_people_min_count:
                if self.multiple_people_start_ts is None:
                    self.multiple_people_start_ts = now
                elif now - self.multiple_people_start_ts >= self.config.multiple_people_seconds:
                    emitted = self._emit_alert(
                        frame,
                        "multiple_people_detected",
                        "high",
                        {
                            "people_count": result.people_count,
                            "min_required": self.config.multiple_people_min_count,
                        },
                        now,
                    )
                    if emitted:
                        active_alerts.append("multiple_people_detected")
            else:
                self.multiple_people_start_ts = None

            if result.offscreen:
                self.offscreen_last_seen_ts = now
                if self.offscreen_start_ts is None:
                    self.offscreen_start_ts = now
                elif now - self.offscreen_start_ts >= self.config.offscreen_seconds:
                    emitted = self._emit_alert(
                        frame,
                        "offscreen_gaze",
                        "medium",
                        {
                            "yaw_ratio": result.debug.get("yaw_ratio", 0.0),
                            "eye_offset": result.debug.get("eye_offset", 0.0),
                            "head_offscreen": result.debug.get("head_offscreen", 0.0),
                            "eye_offscreen": result.debug.get("eye_offscreen", 0.0),
                        },
                        now,
                    )
                    if emitted:
                        active_alerts.append("offscreen_gaze")
            else:
                if self.offscreen_last_seen_ts is None or (now - self.offscreen_last_seen_ts) > 0.5:
                    self.offscreen_start_ts = None

            status = (
                f"Session active | Alerts: {self.alert_count} | "
                f"People: {result.people_count} | Phones: {result.phone_count}"
            )
            if active_alerts:
                status += f" | Last alert: {', '.join(active_alerts)}"

            y_line = 65
            if result.phone_count > 0:
                cv2.putText(
                    frame,
                    "FRAUDE: TELEPHONE DETECTE",
                    (10, y_line),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                y_line += 30
            if result.people_count >= self.config.multiple_people_min_count:
                cv2.putText(
                    frame,
                    "ALERTE: PLUSIEURS PERSONNES",
                    (10, y_line),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                y_line += 30
            if result.offscreen:
                cv2.putText(
                    frame,
                    "ALERTE: REGARD HORS ECRAN (TETE/YEUX)",
                    (10, y_line),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imshow("Anti-Cheat Proctor", frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                print("Arrêt manuel demandé.")
                break

        cap.release()
        cv2.destroyAllWindows()
        self._finalize()

    def _finalize(self) -> None:
        if self.logger is None or self.session_dir is None:
            return
        summary = self.logger.build_summary(
            {
                "student_id": self.config.student_id,
                "subject": self.config.subject,
                "duration_minutes": self.config.duration_minutes,
                "session_dir": str(self.session_dir),
            }
        )
        print("\nRésumé session:")
        print(f"- Total alertes: {summary['total_alerts']}")
        print(f"- Répartition: {summary['alerts_by_type']}")
        print(f"- Dossier session: {self.session_dir}")
