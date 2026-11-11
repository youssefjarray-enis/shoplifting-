"""Execution d'un scenario de bout en bout + calcul des metriques du cahier des charges."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2

from .alert_manager import AlertEvent, AlertManager
from .config import Config
from .deepsort_tracker import SimpleIoUTracker
from .frame_skipper import AdaptiveFrameSkipper
from .pipeline import VisShopPipeline
from .synthetic import Scenario, ScenarioDetector, ScenarioPose, GroundTruth


def build_demo_pipeline(scenario: Scenario, output_dir: str, cfg: Optional[Config] = None
                        ) -> VisShopPipeline:
    cfg = cfg or scenario.make_config()
    alerts = AlertManager(output_dir, cfg.runtime.camera_id, cfg.runtime.save_screenshots)
    skipper = AdaptiveFrameSkipper(cfg.runtime.frame_skip or 3, source_fps=scenario.fps, adaptive=False)
    return VisShopPipeline(cfg, ScenarioDetector(scenario), SimpleIoUTracker(),
                           ScenarioPose(scenario), alerts, skipper=skipper)


@dataclass
class ScenarioRun:
    scenario: Scenario
    events: List[AlertEvent] = field(default_factory=list)
    active_frames: int = 0
    suppressed: Dict[str, int] = field(default_factory=dict)


def run_scenario(scenario: Scenario, output_dir: str, cfg: Optional[Config] = None,
                 video_path: Optional[str] = None) -> ScenarioRun:
    pipe = build_demo_pipeline(scenario, output_dir, cfg)
    run = ScenarioRun(scenario)
    writer = None
    for frame, ts in scenario.frames():
        res = pipe.process_frame(frame, ts)
        run.events += res.events
        run.active_frames += int(any(d.active for d in res.decisions.values()))
        for d in res.decisions.values():
            if d.suppressed_reason:
                run.suppressed[d.suppressed_reason] = run.suppressed.get(d.suppressed_reason, 0) + 1
        if video_path:
            if writer is None:
                h, w = res.frame.shape[:2]
                writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), scenario.fps, (w, h))
            writer.write(res.frame)
    if writer:
        writer.release()
    return run


def compute_metrics(events: List[AlertEvent], truths: List[GroundTruth], duration_s: float,
                    tolerance_s: float = 1.0) -> dict:
    """Un evenement est un vrai positif s'il tombe dans [debut-0.5, fin+tolerance] d'une
    verite terrain (meme comportement). Les doublons sur une meme verite ne sont pas des FP."""
    hit, tp_events, lat = set(), 0, []
    for ev in events:
        matched = None
        for i, gt in enumerate(truths):
            same = gt.behavior is None or gt.behavior.value == ev.behavior
            if same and gt.start - 0.5 <= ev.frame_ts <= gt.end + tolerance_s:
                matched = i
                break
        if matched is not None:
            tp_events += 1
            if matched not in hit:
                lat.append(max(0.0, ev.frame_ts - truths[matched].start))
            hit.add(matched)
    fp = len(events) - tp_events
    fn = len(truths) - len(hit)
    precision = tp_events / len(events) if events else 1.0
    recall = len(hit) / len(truths) if truths else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(events=len(events), true_positives=len(hit), false_positives=fp,
                false_negatives=fn, precision=precision, recall=recall, f1=f1,
                false_positives_per_hour=fp / (duration_s / 3600.0) if duration_s > 0 else 0.0,
                mean_alert_latency_s=sum(lat) / len(lat) if lat else None)


def evaluate_all(names: List[str], output_dir: str) -> dict:
    from .synthetic import build_scenario

    all_events, all_truth, total, per = [], [], 0.0, {}
    offset = 0.0
    for name in names:
        sc = build_scenario(name)
        run = run_scenario(sc, os.path.join(output_dir, name))
        per[name] = dict(compute_metrics(run.events, sc.ground_truth, sc.duration),
                         expected=len(sc.ground_truth), suppressed=run.suppressed)
        for ev in run.events:   # decalage temporel pour agreger sans chevauchement
            ev.frame_ts += offset
        for gt in sc.ground_truth:
            all_truth.append(GroundTruth(gt.start + offset, gt.end + offset, gt.behavior))
        all_events += run.events
        offset += sc.duration + 5.0
        total += sc.duration
    return dict(global_metrics=compute_metrics(all_events, all_truth, total), scenarios=per,
                total_duration_s=total)
