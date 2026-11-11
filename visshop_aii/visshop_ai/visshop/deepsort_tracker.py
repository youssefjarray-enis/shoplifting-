"""Etape 3 : suivi multi-personnes (DeepSORT, avec ReID par features visuelles)."""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from .types import BBox, Detection, Track, iou

log = logging.getLogger(__name__)


class DeepSortTracker:
    """Enveloppe de deep_sort_realtime. ID unique conserve apres occultation partielle
    grace aux embeddings d'apparence (ReID)."""

    def __init__(self, max_age: int = 30, n_init: int = 3, max_cosine_distance: float = 0.4,
                 embedder: str = "mobilenet", use_gpu: bool = True, tracker=None):
        if tracker is None:
            from deep_sort_realtime.deepsort_tracker import DeepSort

            tracker = DeepSort(max_age=max_age, n_init=n_init,
                               max_cosine_distance=max_cosine_distance,
                               embedder=embedder, half=use_gpu,
                               embedder_gpu=use_gpu, bgr=True)
        self.tracker = tracker
        self.max_age = max_age

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        raw = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            raw.append(([x1, y1, x2 - x1, y2 - y1], d.conf, d.label))
        tracks = self.tracker.update_tracks(raw, frame=frame)
        out = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            l, tp, r, b = t.to_ltrb()
            out.append(Track(int(t.track_id), (float(l), float(tp), float(r), float(b)),
                             int(t.time_since_update)))
        return out


class SimpleIoUTracker:
    """Tracker IoU glouton : repli sans dependance (demo, tests, CPU minimal).
    Pas de ReID visuelle : moins robuste que DeepSORT aux croisements."""

    def __init__(self, max_age: int = 30, n_init: int = 2, iou_threshold: float = 0.3):
        self.max_age, self.n_init, self.iou_threshold = max_age, n_init, iou_threshold
        self._tracks: Dict[int, dict] = {}
        self._next_id = 1

    def update(self, detections: List[Detection], frame=None) -> List[Track]:
        det_boxes = [d.bbox for d in detections]
        pairs = []
        for tid, tr in self._tracks.items():
            for j, db in enumerate(det_boxes):
                v = iou(tr["bbox"], db)
                if v >= self.iou_threshold:
                    pairs.append((v, tid, j))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for _, tid, j in pairs:
            if tid in used_t or j in used_d:
                continue
            tr = self._tracks[tid]
            tr.update(bbox=det_boxes[j], missed=0, hits=tr["hits"] + 1)
            used_t.add(tid)
            used_d.add(j)
        for tid, tr in list(self._tracks.items()):
            if tid not in used_t:
                tr["missed"] += 1
                if tr["missed"] > self.max_age:
                    del self._tracks[tid]
        for j, db in enumerate(det_boxes):
            if j not in used_d:
                self._tracks[self._next_id] = dict(bbox=db, missed=0, hits=1)
                self._next_id += 1
        return [Track(tid, tr["bbox"], tr["missed"])
                for tid, tr in self._tracks.items() if tr["hits"] >= self.n_init]


def build_tracker(use_gpu: bool = True, prefer: str = "deepsort"):
    """DeepSORT si installe, sinon repli IoU (avec avertissement)."""
    if prefer == "deepsort":
        try:
            return DeepSortTracker(use_gpu=use_gpu)
        except Exception as exc:
            log.warning("DeepSORT indisponible (%s) -> repli SimpleIoUTracker", exc)
    return SimpleIoUTracker()
