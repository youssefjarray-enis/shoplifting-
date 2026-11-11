"""Etape 4 : analyse de posture - MediaPipe Holistic sur la ROI de chaque personne."""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from .types import BBox, PoseResult

log = logging.getLogger(__name__)


class PoseEstimator:
    """Extrait 33 landmarks corps (+ visibilite) et 21 par main depuis la ROI.

    static_image_mode=True : chaque ROI est une image independante d'une personne
    differente (le mode tracker continu de MediaPipe est incompatible avec
    plusieurs ROIs par frame). La *visibilite* du poignet (0-1) remplace l'axe Z,
    instable en 2D : < 0.25 = main cachee (poche / sac / sous vetement).
    """

    def __init__(self, holistic=None, model_complexity: int = 1,
                 padding: float = 0.10, min_roi: int = 48):
        self.padding, self.min_roi = padding, min_roi
        self._holistic = holistic
        self._complexity = model_complexity

    @property
    def holistic(self):
        if self._holistic is None:
            import mediapipe as mp  # import paresseux

            self._holistic = mp.solutions.holistic.Holistic(
                static_image_mode=True, model_complexity=self._complexity,
                enable_segmentation=False, refine_face_landmarks=False,
                min_detection_confidence=0.5)
        return self._holistic

    def estimate(self, frame: np.ndarray, bbox: BBox) -> Optional[PoseResult]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        px, py = (x2 - x1) * self.padding, (y2 - y1) * self.padding
        x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
        x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
        if x2 - x1 < self.min_roi or y2 - y1 < self.min_roi:
            return None
        roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        res = self.holistic.process(roi)
        if res.pose_landmarks is None:
            return None
        rw, rh = x2 - x1, y2 - y1

        def conv(landmarks, with_vis: bool):
            arr = []
            for lm in landmarks.landmark:
                row = [lm.x * rw + x1, lm.y * rh + y1, lm.z]
                if with_vis:
                    row.append(lm.visibility)
                arr.append(row)
            return np.asarray(arr, dtype=np.float32)

        body = conv(res.pose_landmarks, True)
        lh = conv(res.left_hand_landmarks, False) if res.left_hand_landmarks else None
        rh_ = conv(res.right_hand_landmarks, False) if res.right_hand_landmarks else None
        return PoseResult(body, lh, rh_)

    def close(self) -> None:
        if self._holistic is not None:
            self._holistic.close()
