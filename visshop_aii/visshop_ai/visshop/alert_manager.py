"""Etape 6 : gestion des alertes et journalisation (JSON Lines + captures d'ecran)."""
from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable, Deque, Dict, List, Optional

import cv2
import numpy as np

from .false_alert_filter import FilterDecision
from .types import BBox, Behavior


@dataclass
class AlertEvent:
    timestamp: str
    frame_ts: float
    camera_id: str
    track_id: int
    behavior: str
    behavior_label: str
    score: float
    bbox: List[float]
    screenshot: Optional[str]


@dataclass
class ActiveAlert:
    track_id: int
    behavior: Behavior
    score: float
    bbox: BBox


class AlertManager:
    def __init__(self, output_dir: str = "output", camera_id: str = "cam01",
                 save_screenshots: bool = True,
                 now_fn: Callable[[], datetime] = datetime.now):
        self.output_dir = output_dir
        self.camera_id = camera_id
        self.save_screenshots = save_screenshots
        self.now_fn = now_fn
        self.shots_dir = os.path.join(output_dir, "screenshots")
        os.makedirs(self.shots_dir, exist_ok=True)
        self.log_path = os.path.join(output_dir, "alerts.jsonl")
        self.active_alerts: Dict[int, ActiveAlert] = {}
        self.recent_events: Deque[AlertEvent] = deque(maxlen=5)
        self.total_events = 0

    def handle(self, frame: np.ndarray, decisions: Dict[int, FilterDecision],
               bboxes: Dict[int, BBox], ts: float) -> List[AlertEvent]:
        self.active_alerts = {
            tid: ActiveAlert(tid, d.behavior, d.score, bboxes[tid])
            for tid, d in decisions.items()
            if d.active and d.behavior is not None and tid in bboxes}
        events = []
        for tid, d in decisions.items():
            if d.new_event and tid in bboxes:
                events.append(self._log(frame, d, bboxes[tid], ts))
        return events

    def _log(self, frame, d: FilterDecision, bbox: BBox, ts: float) -> AlertEvent:
        now = self.now_fn()
        shot = None
        if self.save_screenshots:
            img = frame.copy()
            x1, y1, x2, y2 = (int(v) for v in bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(img, f"ID{d.track_id} {d.behavior.label_fr} {d.score:.2f}",
                        (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            shot = os.path.join(
                self.shots_dir,
                f"alert_{now.strftime('%Y%m%d_%H%M%S_%f')}_id{d.track_id}_{d.behavior.value}.jpg")
            cv2.imwrite(shot, img)
        ev = AlertEvent(now.isoformat(timespec="milliseconds"), round(float(ts), 3),
                        self.camera_id, d.track_id, d.behavior.value, d.behavior.label_fr,
                        round(float(d.score), 3), [round(float(v), 1) for v in bbox], shot)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")
        self.recent_events.append(ev)
        self.total_events += 1
        return ev

    @staticmethod
    def read_log(path: str) -> List[dict]:
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
