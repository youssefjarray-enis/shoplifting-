"""Filtre anti-fausses alertes a 4 niveaux.

1. Regles contextuelles : zone rayon + duree de presence, cabine d'essayage
   (detection desactivee), personnel (uniforme) ignore.
2. Lissage temporel : moyenne ponderee sur fenetre glissante de 20 echantillons.
3. Hysteresis : declenchement a 0.68 (lisse) ET score brut >= seuil du comportement ;
   annulation sous 0.35.
4. Cooldown : au plus une alerte journalisee par personne toutes les 3 secondes.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import cv2
import numpy as np

from .config import FilterConfig, StaffConfig, ZoneConfig
from .types import BBox, Behavior, bbox_center, point_in_polygon


# --------------------------------------------------------------------------- #
# Niveau 1 : contexte
# --------------------------------------------------------------------------- #
@dataclass
class ContextInfo:
    in_shelf: bool = False
    shelf_recent: float = 1.0     # 0..1 ; 1.0 si aucune zone rayon configuree
    shelf_visited: bool = True    # visite valide (duree >= min) dans la memoire
    in_fitting_room: bool = False
    is_staff: bool = False


class StaffDetector:
    """Detecte l'uniforme du personnel par couleur HSV dans la zone du torse."""

    def __init__(self, cfg: StaffConfig, ratio: float = 0.35):
        self.ranges = cfg.hsv_ranges
        self.ratio = ratio

    @property
    def enabled(self) -> bool:
        return bool(self.ranges)

    def vote(self, frame: np.ndarray, bbox: BBox) -> Optional[bool]:
        if not self.enabled:
            return None
        x1, y1, x2, y2 = bbox
        h, cx = y2 - y1, (x1 + x2) / 2
        H, W = frame.shape[:2]
        rx1, rx2 = int(max(0, cx - 0.12 * h)), int(min(W, cx + 0.12 * h))
        ry1, ry2 = int(max(0, y1 + 0.22 * h)), int(min(H, y1 + 0.50 * h))
        if rx2 <= rx1 or ry2 <= ry1:
            return None
        hsv = cv2.cvtColor(frame[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in self.ranges:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        return float(mask.mean() / 255.0) >= self.ratio


class ContextAnalyzer:
    def __init__(self, zones: ZoneConfig, cfg: FilterConfig):
        self.zones, self.cfg = zones, cfg
        self._state: Dict[int, dict] = {}

    def set_zones(self, zones: ZoneConfig) -> None:
        self.zones = zones

    def remove_track(self, track_id: int) -> None:
        self._state.pop(track_id, None)

    def update(self, track_id: int, bbox: BBox, ts: float,
               staff_vote: Optional[bool] = None) -> ContextInfo:
        st = self._state.setdefault(track_id, dict(
            entered=None, qualified=None, votes=deque(maxlen=self.cfg.staff_votes_window)))
        foot = ((bbox[0] + bbox[2]) / 2.0, bbox[3])
        in_shelf = any(point_in_polygon(foot, p) for p in self.zones.shelf)
        in_fit = any(point_in_polygon(foot, p) for p in self.zones.fitting_room)

        if in_shelf:
            if st["entered"] is None:
                st["entered"] = ts
            if ts - st["entered"] >= self.cfg.min_shelf_dwell_s:
                st["qualified"] = ts
        else:
            st["entered"] = None

        if not self.zones.shelf:           # pas de zone configuree -> regle neutre
            recent, visited = 1.0, True
        elif st["qualified"] is None:
            recent, visited = 0.0, False
        else:
            age = ts - st["qualified"]
            visited = age <= self.cfg.shelf_memory_s
            recent = float(np.clip(1.0 - age / self.cfg.shelf_memory_s, 0.0, 1.0))

        if staff_vote is not None:
            st["votes"].append(bool(staff_vote))
        v = st["votes"]
        is_staff = len(v) >= 3 and sum(v) / len(v) >= self.cfg.staff_vote_fraction
        return ContextInfo(in_shelf, recent, visited, in_fit, is_staff)


# --------------------------------------------------------------------------- #
# Niveaux 2-4 : lissage, hysteresis, cooldown
# --------------------------------------------------------------------------- #
@dataclass
class FilterDecision:
    track_id: int
    active: bool = False
    behavior: Optional[Behavior] = None
    score: float = 0.0                 # score lisse du comportement actif (ou du meilleur)
    new_event: bool = False            # True uniquement a l'instant ou l'alerte est journalisee
    raw: Dict[Behavior, float] = field(default_factory=dict)
    smoothed: Dict[Behavior, float] = field(default_factory=dict)
    suppressed_reason: Optional[str] = None


class FalseAlertFilter:
    def __init__(self, cfg: FilterConfig, thresholds: Dict[Behavior, float]):
        self.cfg, self.thresholds = cfg, thresholds
        self._s: Dict[int, dict] = {}

    def remove_track(self, track_id: int) -> None:
        self._s.pop(track_id, None)

    def _state(self, tid: int) -> dict:
        return self._s.setdefault(tid, dict(
            win={b: deque(maxlen=self.cfg.smoothing_window) for b in Behavior},
            active=None, last_alert=None, n=0))

    def _smooth(self, win: Deque[float]) -> float:
        if not win:
            return 0.0
        w = np.arange(1, len(win) + 1, dtype=float)   # echantillons recents plus lourds
        return float(np.dot(w, np.asarray(win)) / w.sum())

    def update(self, track_id: int, scores: Dict[Behavior, float], ts: float,
               ctx: ContextInfo) -> FilterDecision:
        st = self._state(track_id)
        reason = None
        if ctx.is_staff:
            reason = "staff"
        elif ctx.in_fitting_room:
            reason = "fitting_room"
        elif not ctx.shelf_visited:
            reason = "no_shelf_visit"

        if reason:   # niveau 1 : on alimente la fenetre avec des zeros et on coupe l'alerte
            scores = {b: 0.0 for b in Behavior}
            st["active"] = None

        for b in Behavior:
            st["win"][b].append(float(scores.get(b, 0.0)))
        st["n"] += 1
        smoothed = {b: self._smooth(st["win"][b]) for b in Behavior}
        new_event = False

        if st["active"] is not None:
            if smoothed[st["active"]] <= self.cfg.cancel_threshold:
                st["active"] = None
        elif not reason and st["n"] >= self.cfg.min_samples:
            cands = []
            for b in Behavior:
                recent_raw = max(list(st["win"][b])[-3:])
                if smoothed[b] >= self.cfg.trigger_threshold and recent_raw >= self.thresholds[b]:
                    cands.append((smoothed[b] / self.thresholds[b], b))
            if cands:
                st["active"] = max(cands)[1]
                last = st["last_alert"]
                if last is None or ts - last >= self.cfg.cooldown_s:
                    new_event, st["last_alert"] = True, ts

        active = st["active"]
        best = active or max(smoothed, key=smoothed.get)
        return FilterDecision(track_id, active is not None, active, smoothed[best],
                              new_event, dict(scores), smoothed, reason)
