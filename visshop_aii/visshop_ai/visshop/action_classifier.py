"""Etape 5 : classification du comportement (moteur de regles + scoring).

Pour chaque personne analysee on calcule 4 scores dans [0,1]. Chaque score est une
somme ponderee de 5 indices, pris sur le meilleur des deux poignets :

  proximite  (0.30) : poignet pres de la cible (poche / buste / sac)
  cache      (0.25) : visibilite du poignet < ~0.25 (+ main non detectee)
  approche   (0.15) : le poignet s'est rapproche de la cible sur ~12 echantillons
  prehension (0.15) : main fermee (objet tenu) dans les 3 dernieres secondes
                      (si aucune main n'est detectee : critere retire, poids renormalises)
  rayon      (0.15) : visite recente du rayon (fournie par le contexte)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np

from .features import SIDES, WRIST_IDX, build_geometry, hand_grip
from .types import BBox, Behavior, PoseResult, distance_point_bbox

HISTORY_LEN = 40
APPROACH_WINDOW = 12
GRIP_MEMORY_S = 3.0
POCKET_RADIUS = 0.5   # en longueurs de buste
TORSO_MARGIN = 0.3
BAG_MARGIN = 0.4
VIS_HIDDEN_START, VIS_HIDDEN_END = 0.60, 0.25

WEIGHTS = {b: (0.30, 0.25, 0.15, 0.15, 0.15) for b in Behavior}  # prox, cache, approche, grip, rayon


@dataclass
class _Sample:
    t: float
    dists: Dict[str, Optional[float]]
    grips: Dict[str, Optional[float]]


def _hidden_score(vis: float, hand_missing: bool) -> float:
    drop = float(np.clip((VIS_HIDDEN_START - vis) / (VIS_HIDDEN_START - VIS_HIDDEN_END), 0, 1))
    return 0.7 * drop + 0.3 * (1.0 if hand_missing else 0.0)


class ActionClassifier:
    def __init__(self, weights=None):
        self.weights = weights or WEIGHTS
        self._hist: Dict[int, Deque[_Sample]] = {}
        self._last: Dict[int, tuple] = {}   # track_id -> (derniers scores, nb de pertes de pose)

    def is_holding(self, track_id: int, ts: float, min_grip: float = 0.5) -> bool:
        """Vrai si une main fermee (objet tenu) a ete vue dans les 3 dernieres secondes."""
        hist = self._hist.get(track_id, ())
        return any(h.t >= ts - GRIP_MEMORY_S and any(g is not None and g >= min_grip for g in h.grips.values())
                   for h in hist)

    def reset_track(self, track_id: int) -> None:
        self._hist.pop(track_id, None)
        self._last.pop(track_id, None)

    def classify(self, track_id: int, pose: Optional[PoseResult], bbox: BBox, ts: float,
                 shelf_recent: float = 1.0,
                 bags: Optional[Dict[str, Optional[BBox]]] = None) -> Dict[Behavior, float]:
        scores = {b: 0.0 for b in Behavior}
        if pose is None:
            # Perte brève de pose (occultation, plan coupé) : on reconduit les derniers scores
            # en les attenuant (x0.9) pendant au plus 5 analyses, puis retour a zero.
            last, miss = self._last.get(track_id, (None, 0))
            if last is not None and miss < 5:
                self._last[track_id] = ({b: v * 0.9 for b, v in last.items()}, miss + 1)
                return dict(self._last[track_id][0])
            return scores
        bags = bags or {}
        geo = build_geometry(pose, bbox)
        L = geo.torso_len

        dists: Dict[str, Optional[float]] = {}
        grips: Dict[str, Optional[float]] = {}
        for s in SIDES:
            w = pose.point(WRIST_IDX[s])
            dists[f"pocket_{s}"] = float(np.hypot(w[0] - geo.pockets[s][0], w[1] - geo.pockets[s][1]) / L)
            dists[f"under_{s}"] = distance_point_bbox(w, geo.torso_box) / L
            for kind in ("backpack", "handbag"):
                bb = bags.get(kind)
                dists[f"{kind}_{s}"] = distance_point_bbox(w, bb) / L if bb is not None else None
            grips[s] = hand_grip(pose.hand(s))

        hist = self._hist.setdefault(track_id, deque(maxlen=HISTORY_LEN))
        hist.append(_Sample(ts, dists, grips))

        def approach(key: str) -> float:
            vals = [h.dists.get(key) for h in list(hist)[-APPROACH_WINDOW:]]
            vals = [v for v in vals if v is not None]
            if len(vals) < 2 or dists.get(key) is None:
                return 0.0
            return float(np.clip((max(vals) - dists[key]) / 0.6, 0, 1))

        def grip_recent(side: str) -> Optional[float]:
            """Meilleure prehension recente, ou None si aucune main n'a ete detectee."""
            g = [h.grips[side] for h in hist if h.t >= ts - GRIP_MEMORY_S and h.grips[side] is not None]
            return max(g) if g else None

        targets = {
            Behavior.POCKET: ("pocket", POCKET_RADIUS),
            Behavior.UNDER_CLOTHING: ("under", TORSO_MARGIN),
            Behavior.BACKPACK: ("backpack", BAG_MARGIN),
            Behavior.HANDBAG: ("handbag", BAG_MARGIN),
        }
        for behavior, (kind, margin) in targets.items():
            wp, wh, wa, wg, ws = self.weights[behavior]
            best = 0.0
            for s in SIDES:
                d = dists[f"{kind}_{s}"]
                if d is None:  # pas de sac associe -> ce comportement n'est pas evaluable
                    continue
                prox = float(np.clip(1.0 - d / margin, 0, 1))
                hidden = _hidden_score(pose.visibility(WRIST_IDX[s]), pose.hand(s) is None)
                g = grip_recent(s)
                shelf = float(np.clip(shelf_recent, 0, 1))
                total = (wp * prox + wh * hidden + wa * approach(f"{kind}_{s}") + ws * shelf)
                if g is None:      # mains jamais detectees (gros plan, basse resolution) :
                    sc = total / (1.0 - wg)   # on retire le critere "prehension" et on renormalise
                else:
                    sc = total + wg * g
                best = max(best, sc)
            scores[behavior] = float(min(best, 1.0))
        self._last[track_id] = (dict(scores), 0)
        return scores
