"""Pipeline complet en 6 etapes (capture -> detection -> suivi -> pose -> classification
-> filtre -> alertes/overlay). Tous les modules sont injectes : chacun peut etre
remplace ou mis a jour independamment."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np

from .action_classifier import ActionClassifier
from .alert_manager import AlertEvent, AlertManager
from .config import Config
from .false_alert_filter import (ContextAnalyzer, FalseAlertFilter, FilterDecision,
                                 StaffDetector)
from .features import BagMemory, associate_bags
from .frame_skipper import AdaptiveFrameSkipper
from .overlay import Incident, MonitorOverlay, Overlay, wrist_crop
from .types import Behavior, Track


@dataclass
class FrameResult:
    frame: np.ndarray
    events: List[AlertEvent] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    decisions: Dict[int, FilterDecision] = field(default_factory=dict)
    fps: float = 0.0
    analyzed: bool = False
    poses: Dict[int, object] = field(default_factory=dict)


class VisShopPipeline:
    def __init__(self, config: Config, detector, tracker, pose_estimator,
                 alert_manager: AlertManager, overlay: Optional[Overlay] = None,
                 skipper: Optional[AdaptiveFrameSkipper] = None,
                 clock=time.perf_counter):
        self.cfg = config
        self.detector, self.tracker, self.pose = detector, tracker, pose_estimator
        self.alerts = alert_manager
        self.overlay = overlay or MonitorOverlay()
        self.skipper = skipper or AdaptiveFrameSkipper(config.runtime.frame_skip or 3)
        self.clock = clock
        self.classifier = ActionClassifier()
        self.context = ContextAnalyzer(config.zones, config.filter)
        self.filter = FalseAlertFilter(config.filter, config.thresholds)
        self.staff = StaffDetector(config.staff, config.filter.staff_ratio)
        self.bag_memory = BagMemory(config.runtime.bag_memory_s)
        self.incidents: List[Incident] = []
        self._poses: Dict[int, object] = {}
        self.frame_idx = -1
        self._zones_ready = False
        self._decisions: Dict[int, FilterDecision] = {}
        self._fps = 0.0

    # ------------------------------------------------------------------ #
    def _forget(self, track_id: int) -> None:
        self._decisions.pop(track_id, None)
        self._poses.pop(track_id, None)
        self.classifier.reset_track(track_id)
        self.context.remove_track(track_id)
        self.filter.remove_track(track_id)
        self.bag_memory.remove_track(track_id)

    def process_frame(self, frame: np.ndarray, ts: float) -> FrameResult:
        t0 = self.clock()
        self.frame_idx += 1
        if not self._zones_ready:
            h, w = frame.shape[:2]
            self.context.set_zones(self.cfg.zones.scaled(w, h))
            self._zones_ready = True

        # Etapes 2-3 : chaque frame
        detections = self.detector.detect(frame)
        persons = [d for d in detections if d.label == "person"]
        bag_dets = [d for d in detections if d.label in ("backpack", "handbag")]
        tracks = self.tracker.update(persons, frame)
        live = {t.track_id for t in tracks}
        for tid in [t for t in self._decisions if t not in live]:
            self._forget(tid)

        # Etapes 4-5 + filtre : 1 frame / N
        fresh: Dict[int, FilterDecision] = {}
        analyzed = self.skipper.should_analyze(self.frame_idx)
        if analyzed:
            bags = self.bag_memory.update(associate_bags(tracks, bag_dets), ts)
            cands = [t for t in tracks if t.time_since_update == 0]
            cands.sort(key=lambda t: -self._decisions[t.track_id].score
                       if t.track_id in self._decisions else 0.0)
            for t in cands[: self.cfg.runtime.max_poses_per_cycle]:
                vote = self.staff.vote(frame, t.bbox)
                ctx = self.context.update(t.track_id, t.bbox, ts, vote)
                pose = self.pose.estimate(frame, t.bbox)
                if pose is not None:
                    self._poses[t.track_id] = pose
                scores = self.classifier.classify(
                    t.track_id, pose, t.bbox, ts, ctx.shelf_recent, bags.get(t.track_id))
                fresh[t.track_id] = self.filter.update(t.track_id, scores, ts, ctx)

        effective: Dict[int, FilterDecision] = {}
        for tid in live:
            if tid in fresh:
                effective[tid] = fresh[tid]
                self._decisions[tid] = replace(fresh[tid], new_event=False)
            elif tid in self._decisions:
                effective[tid] = self._decisions[tid]

        events = self.alerts.handle(frame, effective, {t.track_id: t.bbox for t in tracks}, ts)

        for ev in events:
            beh = Behavior(ev.behavior)
            self.incidents.append(Incident(
                ts, ev.track_id, beh, ev.score,
                wrist_crop(frame, self._poses.get(ev.track_id), tuple(ev.bbox)), tuple(ev.bbox)))
        holding = {t.track_id for t in tracks if self.classifier.is_holding(t.track_id, ts)}
        held = len(holding | set(self.alerts.active_alerts))

        # FPS lisse
        dt = self.clock() - t0
        self.skipper.report(dt)
        inst = 1.0 / dt if dt > 0 else 0.0
        self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst

        stats = dict(camera_id=self.alerts.camera_id, fps=self._fps, persons=len(tracks),
                     total_alerts=self.alerts.total_events,
                     recent=[f"{e.behavior_label} ID{e.track_id}" for e in self.alerts.recent_events],
                     poses=dict(self._poses), holding=holding, held=held, incidents=self.incidents)
        annotated = self.overlay.draw(frame, tracks, self.alerts.active_alerts, stats, ts)
        return FrameResult(annotated, events, tracks, effective, self._fps, analyzed, dict(self._poses))
