"""Types de données et utilitaires géométriques partagés par tous les modules."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels)
Point = Tuple[float, float]


class Behavior(str, Enum):
    """Les 4 comportements de vol détectés."""

    POCKET = "vol_en_poche"
    UNDER_CLOTHING = "cache_sous_vetement"
    BACKPACK = "cache_sac_a_dos"
    HANDBAG = "cache_sac_a_main"

    @property
    def label_fr(self) -> str:
        return {
            Behavior.POCKET: "Vol en poche",
            Behavior.UNDER_CLOTHING: "Cache sous vetement",
            Behavior.BACKPACK: "Cache dans sac a dos",
            Behavior.HANDBAG: "Cache dans sac a main",
        }[self]


class LM:
    """Indices des landmarks corps MediaPipe (33 points) utilisés."""

    NOSE = 0
    L_SHOULDER = 11
    R_SHOULDER = 12
    L_ELBOW = 13
    R_ELBOW = 14
    L_WRIST = 15
    R_WRIST = 16
    L_HIP = 23
    R_HIP = 24
    L_KNEE = 25
    R_KNEE = 26
    L_ANKLE = 27
    R_ANKLE = 28


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    conf: float
    label: str  # "person" | "backpack" | "handbag"


@dataclass
class Track:
    track_id: int
    bbox: BBox
    time_since_update: int = 0  # 0 = vu sur la frame courante, >0 = occulté / prédit
    confirmed: bool = True


@dataclass
class PoseResult:
    """Pose d'une personne en coordonnées pixel de la frame complète.

    body       : (33, 4) -> x, y, z, visibility (visibility dans [0, 1])
    left_hand  : (21, 2|3) ou None si la main n'est pas détectée
    right_hand : idem
    """

    body: np.ndarray
    left_hand: Optional[np.ndarray] = None
    right_hand: Optional[np.ndarray] = None

    def point(self, idx: int) -> Point:
        return float(self.body[idx, 0]), float(self.body[idx, 1])

    def visibility(self, idx: int) -> float:
        return float(self.body[idx, 3])

    def hand(self, side: str) -> Optional[np.ndarray]:
        return self.left_hand if side == "L" else self.right_hand


# --------------------------------------------------------------------------- #
# Géométrie
# --------------------------------------------------------------------------- #
def bbox_center(b: BBox) -> Point:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def bbox_size(b: BBox) -> Tuple[float, float]:
    return b[2] - b[0], b[3] - b[1]


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def expand_bbox(b: BBox, fx: float, fy: float) -> BBox:
    """Agrandit la bbox de fx*largeur (de chaque côté) et fy*hauteur."""
    w, h = bbox_size(b)
    return b[0] - fx * w, b[1] - fy * h, b[2] + fx * w, b[3] + fy * h


def point_in_bbox(p: Point, b: BBox) -> bool:
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def distance_point_bbox(p: Point, b: BBox) -> float:
    """Distance euclidienne du point au rectangle (0 si à l'intérieur)."""
    dx = max(b[0] - p[0], 0.0, p[0] - b[2])
    dy = max(b[1] - p[1], 0.0, p[1] - b[3])
    return float((dx * dx + dy * dy) ** 0.5)


def point_in_polygon(p: Point, poly: Sequence[Sequence[float]]) -> bool:
    """Test d'appartenance à un polygone (ray casting)."""
    n = len(poly)
    if n < 3:
        return False
    x, y = p
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside
