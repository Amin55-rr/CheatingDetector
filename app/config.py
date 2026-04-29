from dataclasses import dataclass


@dataclass
class SessionConfig:
    student_id: str
    subject: str
    duration_minutes: int
    webcam_index: int = 0
    confidence_threshold: float = 0.45
    # Durée de regard "hors écran" (tête ou yeux) avant alerte — modéré, pas agressif
    offscreen_seconds: float = 0.6
    capture_on_alert: bool = True
    # Évite de spammer les mêmes alertes
    alert_cooldown_seconds: float = 2.5
    multiple_people_min_count: int = 2
    multiple_people_seconds: float = 1.0
