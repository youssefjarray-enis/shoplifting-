"""Scenarios synthetiques deterministes (sans camera, sans modele, sans GPU).

Chaque scenario decrit des personnes (keyframes : position + poignets + visibilite +
prehension) et produit frames, detections et poses. Sert a : demo (`main.py --demo`),
tests d'integration et rapport de metriques.
Unites des poignets : fraction de la hauteur H, relative a (centre x, sommet de la tete).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, fields
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .config import Config, StaffConfig, ZoneConfig
from .types import BBox, Behavior, Detection, LM, PoseResult, iou

Pt = Tuple[float, float]
HANG_L, HANG_R = (-0.16, 0.50), (0.16, 0.50)
REACH_R = (0.25, 0.30)
POCKET_R = (0.08, 0.58)
BELLY_R = (0.03, 0.40)
HANDBAG_WRIST_R = (0.24, 0.54)
BACKPACK_WRIST_R = (0.22, 0.34)
HANDBAG_BOX = (0.17, 0.42, 0.32, 0.66)
BACKPACK_BOX = (0.12, 0.18, 0.30, 0.50)
STAFF_SHIRT = (200, 100, 30)      # BGR bleu -> HSV ~ (107, 217, 200)
CUSTOMER_SHIRT = (40, 40, 200)    # BGR rouge
FEET_Y = 500.0


@dataclass
class Key:
    t: float
    cx: float
    fy: float = FEET_Y
    lw: Pt = HANG_L
    rw: Pt = HANG_R
    lvis: float = 0.95
    rvis: float = 0.95
    lgrip: float = 0.0
    rgrip: float = 0.0


def _lerp(a, b, u):
    if isinstance(a, tuple):
        return tuple(x + (y - x) * u for x, y in zip(a, b))
    return a + (b - a) * u


def interpolate(a: Key, b: Key, u: float) -> Key:
    return Key(**{f.name: _lerp(getattr(a, f.name), getattr(b, f.name), u) for f in fields(Key)})


@dataclass
class BagSpec:
    kind: str                    # "backpack" | "handbag"
    rel_box: Tuple[float, float, float, float]


@dataclass
class Actor:
    name: str
    keys: List[Key]
    shirt: Tuple[int, int, int] = CUSTOMER_SHIRT
    height: float = 320.0
    bags: List[BagSpec] = field(default_factory=list)

    def key_at(self, t: float) -> Key:
        ks = self.keys
        if t <= ks[0].t:
            return ks[0]
        for a, b in zip(ks, ks[1:]):
            if a.t <= t <= b.t:
                return interpolate(a, b, (t - a.t) / max(b.t - a.t, 1e-9))
        return ks[-1]

    def bbox_at(self, t: float) -> BBox:
        k, H = self.key_at(t), self.height
        return (k.cx - 0.22 * H, k.fy - H, k.cx + 0.22 * H, k.fy)

    def pose_at(self, t: float) -> PoseResult:
        k, H = self.key_at(t), self.height
        top = k.fy - H
        body = np.zeros((33, 4), np.float32)
        body[:, 0], body[:, 1], body[:, 3] = k.cx, top + 0.5 * H, 0.9

        def put(idx, dx, dy, vis=0.95):
            body[idx] = (k.cx + dx * H, top + dy * H, 0.0, vis)

        put(LM.NOSE, 0, 0.06)
        put(LM.L_SHOULDER, -0.13, 0.20); put(LM.R_SHOULDER, 0.13, 0.20)
        put(LM.L_HIP, -0.08, 0.52); put(LM.R_HIP, 0.08, 0.52)
        put(LM.L_KNEE, -0.08, 0.74); put(LM.R_KNEE, 0.08, 0.74)
        put(LM.L_ANKLE, -0.08, 0.95); put(LM.R_ANKLE, 0.08, 0.95)
        put(LM.L_WRIST, *k.lw, vis=k.lvis); put(LM.R_WRIST, *k.rw, vis=k.rvis)
        put(LM.L_ELBOW, (-0.13 + k.lw[0]) / 2, (0.20 + k.lw[1]) / 2 + 0.04, 0.9)
        put(LM.R_ELBOW, (0.13 + k.rw[0]) / 2, (0.20 + k.rw[1]) / 2 + 0.04, 0.9)
        lh = make_hand(body[LM.L_WRIST, 0], body[LM.L_WRIST, 1], 0.06 * H, k.lgrip) if k.lvis >= 0.5 else None
        rh = make_hand(body[LM.R_WRIST, 0], body[LM.R_WRIST, 1], 0.06 * H, k.rgrip) if k.rvis >= 0.5 else None
        return PoseResult(body, lh, rh)

    def bag_dets_at(self, t: float) -> List[Detection]:
        k, H = self.key_at(t), self.height
        top = k.fy - H
        return [Detection((k.cx + b.rel_box[0] * H, top + b.rel_box[1] * H,
                           k.cx + b.rel_box[2] * H, top + b.rel_box[3] * H), 0.9, b.kind)
                for b in self.bags]


def make_hand(cx: float, cy: float, size: float, grip: float) -> np.ndarray:
    """21 landmarks synthetiques : main ouverte (grip=0) -> poing ferme (grip=1)."""
    pts = np.zeros((21, 2), np.float32)
    pts[0] = (cx, cy)
    xs = [-0.3, -0.1, 0.1, 0.3]
    for k, base in enumerate([5, 9, 13, 17]):
        mcp = np.array([cx + xs[k] * size, cy - size])
        tip_y = cy - size - 0.9 * size * (1 - grip) + 0.1 * size * grip
        pts[base] = mcp
        pts[base + 3] = (mcp[0], tip_y)
        pts[base + 1] = mcp + (pts[base + 3] - mcp) * 0.4
        pts[base + 2] = mcp + (pts[base + 3] - mcp) * 0.75
    palm = np.array([cx, cy - 0.8 * size])
    thumb_tip = np.array([cx - 0.7 * size, cy - 0.7 * size]) * (1 - grip) + palm * grip
    for i, f in zip((1, 2, 3, 4), (0.25, 0.5, 0.75, 1.0)):
        pts[i] = pts[0] + (thumb_tip - pts[0]) * f
    return pts


@dataclass
class GroundTruth:
    start: float
    end: float
    behavior: Optional[Behavior] = None


@dataclass
class Scenario:
    name: str
    duration: float
    actors: List[Actor]
    ground_truth: List[GroundTruth]
    zones: ZoneConfig
    description: str = ""
    fps: float = 30.0
    size: Tuple[int, int] = (960, 540)
    cursor: float = 0.0          # temps courant (lu par ScenarioDetector / ScenarioPose)

    def __post_init__(self):
        self.zones = copy.deepcopy(self.zones)

    @property
    def n_frames(self) -> int:
        return int(self.duration * self.fps)

    def make_config(self) -> Config:
        cfg = Config()
        cfg.zones = copy.deepcopy(self.zones)   # evite de partager l'etat entre scenarios
        cfg.staff = StaffConfig(hsv_ranges=[((100, 100, 50), (130, 255, 255))])
        cfg.runtime.frame_skip = 3
        cfg.runtime.adaptive_skip = False
        return cfg

    def frames(self) -> Iterator[Tuple[np.ndarray, float]]:
        for i in range(self.n_frames):
            t = i / self.fps
            self.cursor = t
            yield self.render(t), t

    def detections_at(self, t: float) -> List[Detection]:
        out = []
        for a in self.actors:
            out.append(Detection(a.bbox_at(t), 0.95, "person"))
            out += a.bag_dets_at(t)
        return out

    def render(self, t: float) -> np.ndarray:
        w, h = self.size
        img = np.full((h, w, 3), (70, 70, 70), np.uint8)
        cv2.rectangle(img, (500, 150), (920, 380), (40, 70, 110), -1)   # rayonnages
        for poly, col in [(p, (0, 140, 0)) for p in self.zones.shelf] + \
                         [(p, (140, 80, 0)) for p in self.zones.fitting_room]:
            cv2.polylines(img, [np.array(poly, np.int32)], True, col, 1)
        for a in self.actors:
            self._draw_actor(img, a, t)
        return img

    @staticmethod
    def _draw_actor(img, a: Actor, t: float) -> None:
        pose, H = a.pose_at(t), a.height
        P = lambda i: (int(pose.body[i, 0]), int(pose.body[i, 1]))
        for bag in a.bag_dets_at(t):
            x1, y1, x2, y2 = (int(v) for v in bag.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (120, 60, 140) if bag.label == "handbag" else (30, 90, 60), -1)
        cv2.circle(img, P(LM.NOSE), int(0.05 * H), (170, 190, 230), -1)
        cv2.rectangle(img, (P(LM.L_SHOULDER)[0], P(LM.L_SHOULDER)[1]),
                      (P(LM.R_HIP)[0], P(LM.R_HIP)[1]), a.shirt, -1)
        for i, j in [(LM.L_HIP, LM.L_ANKLE), (LM.R_HIP, LM.R_ANKLE), (LM.L_SHOULDER, LM.L_ELBOW),
                     (LM.L_ELBOW, LM.L_WRIST), (LM.R_SHOULDER, LM.R_ELBOW), (LM.R_ELBOW, LM.R_WRIST)]:
            cv2.line(img, P(i), P(j), (90, 90, 160), 6)
        for idx in (LM.L_WRIST, LM.R_WRIST):
            if pose.body[idx, 3] >= 0.5:
                cv2.circle(img, P(idx), 7, (170, 190, 230), -1)


class ScenarioDetector:
    def __init__(self, scenario: Scenario):
        self.s = scenario

    def detect(self, frame) -> List[Detection]:
        return self.s.detections_at(self.s.cursor)


class ScenarioPose:
    def __init__(self, scenario: Scenario):
        self.s = scenario

    def estimate(self, frame, bbox: BBox) -> Optional[PoseResult]:
        best, best_iou = None, 0.3
        for a in self.s.actors:
            v = iou(a.bbox_at(self.s.cursor), bbox)
            if v > best_iou:
                best, best_iou = a, v
        return best.pose_at(self.s.cursor) if best else None


# --------------------------------------------------------------------------- #
# Construction des scenarios
# --------------------------------------------------------------------------- #
ZONES = ZoneConfig(
    shelf=[[[500, 380], [920, 380], [920, 540], [500, 540]]],
    fitting_room=[[[30, 380], [250, 380], [250, 540], [30, 540]]])


def _theft_keys(conceal: Pt, vis: float = 0.1, cx0=250, cx_shelf=600, cx_end=850,
                t_move=4.8, t_hidden=5.3, t_release=7.5, t_back=8.0, T=10.5) -> List[Key]:
    return [Key(0, cx0), Key(2.0, cx_shelf),
            Key(3.0, cx_shelf, rw=REACH_R),
            Key(3.5, cx_shelf, rw=REACH_R, rgrip=0.9),
            Key(t_move, cx_shelf, rw=(0.2, 0.30), rgrip=0.9),
            Key(t_hidden, cx_shelf, rw=conceal, rvis=vis, rgrip=0.9),
            Key(t_release, cx_shelf, rw=conceal, rvis=vis, rgrip=0.9),
            Key(t_back, cx_shelf),
            Key(T, cx_end)]


def _browse_keys(cx0=250, cx_shelf=600, cx_end=850, T=10.5) -> List[Key]:
    return [Key(0, cx0), Key(2.0, cx_shelf), Key(3.0, cx_shelf, rw=REACH_R),
            Key(3.5, cx_shelf, rw=REACH_R, rgrip=0.9),
            Key(5.0, cx_shelf, rw=(0.22, 0.32), rgrip=0.9),     # regarde le produit
            Key(5.5, cx_shelf, rw=REACH_R, rgrip=0.0),           # le repose
            Key(6.5, cx_shelf), Key(T, cx_end)]


def build_scenario(name: str) -> Scenario:
    gt_t = lambda a, b, beh: [GroundTruth(a, b, beh)]
    T = 10.5
    if name == "theft_pocket":
        a = Actor("thief", _theft_keys(POCKET_R))
        return Scenario(name, T, [a], gt_t(5.3, 7.5, Behavior.POCKET), ZONES,
                        "Le client prend un produit au rayon et le glisse dans sa poche.")
    if name == "theft_under_clothing":
        a = Actor("thief", _theft_keys(BELLY_R))
        return Scenario(name, T, [a], gt_t(5.3, 7.5, Behavior.UNDER_CLOTHING), ZONES,
                        "Le produit est dissimule sous la veste.")
    if name == "theft_backpack":
        a = Actor("thief", _theft_keys(BACKPACK_WRIST_R), bags=[BagSpec("backpack", BACKPACK_BOX)])
        return Scenario(name, T, [a], gt_t(5.3, 7.5, Behavior.BACKPACK), ZONES,
                        "Le produit est insere dans un sac a dos.")
    if name == "theft_handbag":
        a = Actor("thief", _theft_keys(HANDBAG_WRIST_R), bags=[BagSpec("handbag", HANDBAG_BOX)])
        return Scenario(name, T, [a], gt_t(5.3, 7.5, Behavior.HANDBAG), ZONES,
                        "Le produit est glisse dans un sac a main.")
    if name == "normal_browse":
        return Scenario(name, T, [Actor("customer", _browse_keys())], [], ZONES,
                        "Le client prend un produit, le regarde, le repose.")
    if name == "normal_stow_own_items":
        keys = [Key(0, 320), Key(2.0, 350, rw=(0.2, 0.45), rgrip=0.9),
                Key(3.0, 350, rw=(0.2, 0.45), rgrip=0.9),
                Key(3.5, 350, rw=HANDBAG_WRIST_R, rvis=0.1, rgrip=0.9),
                Key(6.0, 350, rw=HANDBAG_WRIST_R, rvis=0.1, rgrip=0.9),
                Key(6.5, 350), Key(T, 330)]
        a = Actor("customer", keys, bags=[BagSpec("handbag", HANDBAG_BOX)])
        return Scenario(name, T, [a], [], ZONES,
                        "Le client range ses propres affaires dans son sac, sans passer au rayon.")
    if name == "normal_fitting_room":
        keys = [Key(0, 400), Key(2.0, 600), Key(3.0, 600, rw=REACH_R),
                Key(3.5, 600, rw=REACH_R, rgrip=0.9),
                Key(4.5, 600, rw=(0.2, 0.45), rgrip=0.9),
                Key(7.0, 140, rw=(0.2, 0.45), rgrip=0.9),
                Key(7.8, 140, rw=POCKET_R, rvis=0.1, rgrip=0.9),
                Key(10.0, 140, rw=POCKET_R, rvis=0.1, rgrip=0.9),
                Key(10.5, 140), Key(11.5, 140)]
        return Scenario(name, 11.5, [Actor("customer", keys)], [], ZONES,
                        "Le client essaie un vetement en cabine d'essayage (detection desactivee).")
    if name == "normal_staff_restock":
        a = Actor("staff", _theft_keys(POCKET_R), shirt=STAFF_SHIRT)
        return Scenario(name, T, [a], [], ZONES,
                        "Un employe en uniforme reapprovisionne le rayon (gestes similaires).")
    if name == "normal_quick_gesture":
        a = Actor("customer", _theft_keys(POCKET_R, t_move=4.8, t_hidden=5.0, t_release=5.4, t_back=5.6))
        return Scenario(name, T, [a], [], ZONES,
                        "Geste rapide inoffensif (main a la poche 0.4 s : se gratte la hanche).")
    if name == "two_customers":
        thief = Actor("thief", _theft_keys(POCKET_R))
        other = Actor("customer", _browse_keys(cx0=920, cx_shelf=830, cx_end=900))
        return Scenario(name, T, [thief, other], gt_t(5.3, 7.5, Behavior.POCKET), ZONES,
                        "Deux clients au rayon : un voleur, un client normal.")
    raise KeyError(f"Scenario inconnu: {name}")


SCENARIO_NAMES = ["theft_pocket", "theft_under_clothing", "theft_backpack", "theft_handbag",
                  "normal_browse", "normal_stow_own_items", "normal_fitting_room",
                  "normal_staff_restock", "normal_quick_gesture", "two_customers"]
