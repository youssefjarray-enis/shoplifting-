"""Extraction de caracteristiques geometriques a partir de la pose 2D."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .types import BBox, LM, Point, Detection, Track, bbox_center, expand_bbox, point_in_bbox, PoseResult

SIDES = ("L", "R")
WRIST_IDX = {"L": LM.L_WRIST, "R": LM.R_WRIST}
MIN_LANDMARK_VIS = 0.35


@dataclass
class Geometry:
    torso_len: float
    shoulder_half_w: float
    pockets: Dict[str, Point]   # centre de la poche de chaque cote
    torso_box: BBox             # zone "sous le vetement"


def build_geometry(pose: PoseResult, bbox: BBox) -> Geometry:
    """Geometrie du buste, robuste aux gros plans.

    Les hanches ne sont utilisees que si elles sont visibles ET dans la bbox de la personne
    (MediaPipe "invente" des hanches hors cadre sur un plan serre, avec une visibilite
    elevee). Sinon la longueur du buste est deduite de la largeur des epaules (~1.3x).
    Si les epaules sont elles-memes invisibles : proportions de la bbox (corps entier)."""
    x1, y1, x2, y2 = bbox
    h = max(y2 - y1, 1.0)
    cx = (x1 + x2) / 2.0

    def ok(idx, inside_bbox=False):
        if pose.visibility(idx) < MIN_LANDMARK_VIS:
            return False
        return (not inside_bbox) or pose.point(idx)[1] <= y2 + 0.05 * h

    if ok(LM.L_SHOULDER) and ok(LM.R_SHOULDER):
        ls, rs = pose.point(LM.L_SHOULDER), pose.point(LM.R_SHOULDER)
    else:
        ls, rs = (cx - 0.13 * h, y1 + 0.20 * h), (cx + 0.13 * h, y1 + 0.20 * h)
    sh_y = (ls[1] + rs[1]) / 2
    half_w = max(abs(rs[0] - ls[0]) / 2, 0.05 * h)
    tx = (ls[0] + rs[0]) / 2

    if ok(LM.L_HIP, True) and ok(LM.R_HIP, True):
        lh, rh = pose.point(LM.L_HIP), pose.point(LM.R_HIP)
        hip_y = (lh[1] + rh[1]) / 2
        torso_len = max(hip_y - sh_y, 0.1 * h)
    else:  # plan serre : hanches estimees a partir de la largeur des epaules
        torso_len = 1.3 * (2 * half_w)
        hip_y = sh_y + torso_len
        lh, rh = (tx - 0.6 * half_w, hip_y), (tx + 0.6 * half_w, hip_y)

    pockets = {"L": (lh[0], lh[1] + 0.18 * torso_len),
               "R": (rh[0], rh[1] + 0.18 * torso_len)}
    torso_box = (tx - 0.9 * half_w, sh_y + 0.15 * torso_len,
                 tx + 0.9 * half_w, hip_y + 0.05 * torso_len)
    return Geometry(torso_len, half_w, pockets, torso_box)


def hand_grip(hand: Optional[np.ndarray]) -> Optional[float]:
    """Score de prehension [0,1] : bouts des doigts proches de la paume = main fermee
    (objet tenu). None si la main n'est pas detectee."""
    if hand is None or len(hand) < 21:
        return None
    h = np.asarray(hand, dtype=float)[:, :2]
    size = float(np.linalg.norm(h[9] - h[0]))
    if size < 1e-6:
        return None
    palm = h[[0, 5, 9, 13, 17]].mean(axis=0)
    tips = h[[4, 8, 12, 16, 20]]
    ratio = float(np.linalg.norm(tips - palm, axis=1).mean() / size)
    return float(np.clip((1.0 - ratio) / 0.5, 0.0, 1.0))


def associate_bags(tracks: List[Track], bag_dets: List[Detection]) -> Dict[int, Dict[str, Optional[BBox]]]:
    """Associe chaque sac detecte a la personne la plus proche dont la bbox
    (elargie de 35% en largeur, 10% en hauteur) contient le centre du sac."""
    out: Dict[int, Dict[str, Optional[BBox]]] = {
        t.track_id: {"backpack": None, "handbag": None} for t in tracks}
    best_conf: Dict[tuple, float] = {}
    for bag in bag_dets:
        c = bbox_center(bag.bbox)
        best, best_d = None, None
        for t in tracks:
            if not point_in_bbox(c, expand_bbox(t.bbox, 0.35, 0.10)):
                continue
            tc = bbox_center(t.bbox)
            d = (c[0] - tc[0]) ** 2 + (c[1] - tc[1]) ** 2
            if best_d is None or d < best_d:
                best, best_d = t, d
        if best is None:
            continue
        key = (best.track_id, bag.label)
        if bag.conf >= best_conf.get(key, -1.0):
            out[best.track_id][bag.label] = bag.bbox
            best_conf[key] = bag.conf
    return out


class BagMemory:
    """Garde le dernier sac vu de chaque personne quelques instants : YOLO rate souvent un
    sac partiellement cache par le bras ou le corps, ce qui ferait tomber le score a 0."""

    def __init__(self, memory_s: float = 1.5):
        self.memory_s = memory_s
        self._last: Dict[tuple, tuple] = {}   # (track_id, kind) -> (bbox, ts)

    def update(self, bags: Dict[int, Dict[str, Optional[BBox]]], ts: float
               ) -> Dict[int, Dict[str, Optional[BBox]]]:
        out: Dict[int, Dict[str, Optional[BBox]]] = {}
        for tid, kinds in bags.items():
            out[tid] = {}
            for kind, bb in kinds.items():
                if bb is not None:
                    self._last[(tid, kind)] = (bb, ts)
                    out[tid][kind] = bb
                else:
                    prev = self._last.get((tid, kind))
                    out[tid][kind] = prev[0] if prev and ts - prev[1] <= self.memory_s else None
        return out

    def remove_track(self, track_id: int) -> None:
        for k in [k for k in self._last if k[0] == track_id]:
            del self._last[k]
