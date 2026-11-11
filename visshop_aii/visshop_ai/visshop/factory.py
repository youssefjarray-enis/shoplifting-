"""Assemblage du pipeline reel (YOLO + DeepSORT + MediaPipe)."""
from __future__ import annotations

from typing import Optional

from .alert_manager import AlertManager
from .config import Config
from .deepsort_tracker import build_tracker
from .frame_skipper import AdaptiveFrameSkipper
from .model_selector import select_model
from .pipeline import VisShopPipeline
from .pose_estimator import PoseEstimator
from .yolo_detector import YoloDetector


def build_real_pipeline(cfg: Config, output_dir: str = "output", device: Optional[str] = None,
                        source_fps: float = 30.0, adaptive: Optional[bool] = None) -> VisShopPipeline:
    choice = select_model(force=device)
    detector = YoloDetector(choice, conf=cfg.runtime.detection_conf)
    tracker = build_tracker(use_gpu=choice.is_gpu)
    pose = PoseEstimator()
    skipper = AdaptiveFrameSkipper(
        cfg.runtime.frame_skip or choice.frame_skip, source_fps=source_fps,
        adaptive=cfg.runtime.adaptive_skip if adaptive is None else adaptive)
    alerts = AlertManager(output_dir, cfg.runtime.camera_id, cfg.runtime.save_screenshots)
    return VisShopPipeline(cfg, detector, tracker, pose, alerts, skipper=skipper)
