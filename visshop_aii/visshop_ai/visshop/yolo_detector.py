"""Etape 2 : detection des personnes (+ sacs) avec YOLOv11n / YOLOv8n (Ultralytics)."""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from .model_selector import ModelChoice, select_model
from .types import Detection

log = logging.getLogger(__name__)

# Classes COCO utiles : personne, sac a dos, sac a main
COCO_CLASSES = {0: "person", 24: "backpack", 26: "handbag"}


def _to_np(x) -> np.ndarray:
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


class YoloDetector:
    """Detecte personnes + sacs. Bascule automatiquement sur le modele fallback si le
    modele principal ne peut pas etre charge. Un `model` peut etre injecte (tests)."""

    def __init__(self, choice: Optional[ModelChoice] = None, conf: float = 0.4,
                 iou: float = 0.5, imgsz: int = 640, model=None):
        self.choice = choice or select_model()
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.model = model if model is not None else self._load()

    def _load(self):
        from ultralytics import YOLO  # import paresseux

        for weights in (self.choice.weights, self.choice.fallback_weights):
            try:
                log.info("Chargement du modele %s sur %s", weights, self.choice.device)
                return YOLO(weights)
            except Exception as exc:
                log.warning("Echec chargement %s: %s", weights, exc)
        raise RuntimeError("Aucun modele YOLO n'a pu etre charge")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            frame, classes=list(COCO_CLASSES), conf=self.conf, iou=self.iou,
            imgsz=self.imgsz, device=self.choice.device, half=self.choice.half,
            verbose=False,
        )
        return self.parse_results(results)

    @staticmethod
    def parse_results(results) -> List[Detection]:
        out: List[Detection] = []
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None:
            return out
        xyxy, conf, cls = _to_np(boxes.xyxy), _to_np(boxes.conf), _to_np(boxes.cls)
        for b, c, k in zip(xyxy, conf, cls):
            label = COCO_CLASSES.get(int(k))
            if label:
                out.append(Detection(tuple(float(v) for v in b), float(c), label))
        return out
