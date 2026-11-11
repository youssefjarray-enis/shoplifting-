"""Etapes 1-2 : selection automatique du modele de detection (YOLOv11n / YOLOv8n)."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

MIN_VRAM_GB = 5.5  # GTX 1660 = 6 Go nominal (~5.9 Go reportes)


@dataclass(frozen=True)
class DeviceInfo:
    has_cuda: bool
    vram_gb: float = 0.0
    name: str = "cpu"


@dataclass(frozen=True)
class ModelChoice:
    name: str          # "yolo11n" | "yolov8n"
    weights: str       # fichier .pt (telecharge automatiquement par Ultralytics)
    fallback_weights: str
    device: str        # "cuda:0" | "cpu"
    half: bool
    frame_skip: int    # N : MediaPipe tourne toutes les N frames

    @property
    def is_gpu(self) -> bool:
        return self.device.startswith("cuda")


def detect_device() -> DeviceInfo:
    """Detecte le GPU NVIDIA (import torch paresseux : fonctionne sans torch)."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return DeviceInfo(True, props.total_memory / 1024 ** 3, props.name)
    except Exception as exc:  # torch absent ou CUDA casse
        log.debug("Detection GPU impossible: %s", exc)
    return DeviceInfo(False)


def select_model(device_info: Optional[DeviceInfo] = None, force: Optional[str] = None) -> ModelChoice:
    """Choisit modele + frame skipping.

    GPU >= 6 Go  -> YOLOv11n (principal), half precision, MediaPipe 1 frame / 3
    sinon (CPU)  -> YOLOv8n (fallback auto), Frame Skip x5
    `force` ("cpu" | "cuda") ou variable d'environnement VISSHOP_DEVICE force le choix.
    """
    force = force or os.environ.get("VISSHOP_DEVICE")
    info = device_info or detect_device()

    use_gpu = info.has_cuda and info.vram_gb >= MIN_VRAM_GB
    if info.has_cuda and not use_gpu:
        log.warning("GPU %s (%.1f Go) sous le minimum requis (6 Go) -> mode CPU",
                    info.name, info.vram_gb)
    if force == "cpu":
        use_gpu = False
    elif force and force.startswith("cuda"):
        use_gpu = info.has_cuda

    if use_gpu:
        return ModelChoice("yolo11n", "yolo11n.pt", "yolov8n.pt", "cuda:0", True, 3)
    return ModelChoice("yolov8n", "yolov8n.pt", "yolov8n.pt", "cpu", False, 5)


if __name__ == "__main__":
    print(select_model())
