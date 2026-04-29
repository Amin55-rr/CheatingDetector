import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class AlertLogger:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.captures_dir = session_dir / "captures"
        self.log_file = session_dir / "alerts.jsonl"
        self.summary_file = session_dir / "summary.json"
        self.captures_dir.mkdir(parents=True, exist_ok=True)
        self._events: List[Dict[str, Any]] = []
    def add_event(self, event_type: str, severity: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        event = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "type": event_type,
            "severity": severity,
            "meta": meta,
        }
        self._events.append(event)
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def build_summary(self, session_info: Dict[str, Any]) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        for e in self._events:
            by_type[e["type"]] = by_type.get(e["type"], 0) + 1
        summary = {
            "session": session_info,
            "total_alerts": len(self._events),
            "alerts_by_type": by_type,
            "events": self._events,
        }
        with self.summary_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary
