"""Tests de bout en bout sur les scenarios synthetiques (pipeline complet, sans modele)."""
import os

import pytest

from visshop.alert_manager import AlertManager
from visshop.evaluation import compute_metrics, evaluate_all, run_scenario
from visshop.synthetic import SCENARIO_NAMES, GroundTruth, build_scenario
from visshop.types import Behavior

THEFTS = {"theft_pocket": Behavior.POCKET, "theft_under_clothing": Behavior.UNDER_CLOTHING,
          "theft_backpack": Behavior.BACKPACK, "theft_handbag": Behavior.HANDBAG,
          "two_customers": Behavior.POCKET}
NORMALS = [n for n in SCENARIO_NAMES if n.startswith("normal_")]


@pytest.mark.parametrize("name,behavior", THEFTS.items())
def test_theft_detected_with_right_behavior_and_latency(name, behavior, tmp_path):
    sc = build_scenario(name)
    run = run_scenario(sc, str(tmp_path))
    assert len(run.events) == 1, run.events
    ev = run.events[0]
    gt = sc.ground_truth[0]
    assert ev.behavior == behavior.value
    assert gt.start - 0.5 <= ev.frame_ts <= gt.end
    assert ev.frame_ts - gt.start < 2.5                       # latence d'alerte < 2.5 s
    assert os.path.exists(ev.screenshot)
    assert len(AlertManager.read_log(os.path.join(str(tmp_path), "alerts.jsonl"))) == 1
    assert 0.68 <= ev.score <= 1.0


@pytest.mark.parametrize("name", NORMALS)
def test_normal_behaviour_raises_no_alert(name, tmp_path):
    run = run_scenario(build_scenario(name), str(tmp_path))
    assert run.events == [] and run.active_frames == 0


def test_rules_actually_do_the_suppression(tmp_path):
    assert run_scenario(build_scenario("normal_fitting_room"), str(tmp_path / "a")).suppressed.get("fitting_room", 0) > 0
    assert run_scenario(build_scenario("normal_staff_restock"), str(tmp_path / "b")).suppressed.get("staff", 0) > 0
    assert run_scenario(build_scenario("normal_stow_own_items"), str(tmp_path / "c")).suppressed.get("no_shelf_visit", 0) > 0


def test_without_context_rules_the_same_gestures_would_alert(tmp_path):
    """Preuve que les regles contextuelles sont necessaires : sans zones ni uniforme,
    le personnel et la cabine d'essayage declencheraient des fausses alertes."""
    for name in ("normal_staff_restock", "normal_fitting_room"):
        sc = build_scenario(name)
        cfg = sc.make_config()
        cfg.zones.shelf, cfg.zones.fitting_room, cfg.staff.hsv_ranges = [], [], []
        assert len(run_scenario(sc, str(tmp_path / name), cfg).events) >= 1


def test_alert_ends_after_gesture(tmp_path):
    sc = build_scenario("theft_pocket")
    run = run_scenario(sc, str(tmp_path))
    assert 0 < run.active_frames < sc.n_frames * 0.5


def test_thief_bbox_is_the_one_alerted_with_two_people(tmp_path):
    sc = build_scenario("two_customers")
    ev = run_scenario(sc, str(tmp_path)).events[0]
    cx = (ev.bbox[0] + ev.bbox[2]) / 2
    assert abs(cx - 600) < 40                                   # le voleur (x=600), pas le client (x~830)


def test_global_metrics_meet_spec_targets(tmp_path):
    rep = evaluate_all(SCENARIO_NAMES, str(tmp_path))
    g = rep["global_metrics"]
    assert g["precision"] >= 0.92 and g["recall"] >= 0.87 and g["f1"] >= 0.89
    assert g["false_positives_per_hour"] < 2
    assert g["mean_alert_latency_s"] < 2.5


def test_compute_metrics_arithmetic():
    from visshop.alert_manager import AlertEvent
    mk = lambda ts, b="vol_en_poche": AlertEvent("t", ts, "c", 1, b, "x", 0.8, [0, 0, 1, 1], None)
    gts = [GroundTruth(10, 12, Behavior.POCKET), GroundTruth(30, 32, Behavior.POCKET)]
    m = compute_metrics([mk(11), mk(50)], gts, 3600)
    assert (m["true_positives"], m["false_positives"], m["false_negatives"]) == (1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["false_positives_per_hour"] == 1.0
    assert compute_metrics([mk(11, "cache_sac_a_main")], gts[:1], 3600)["false_positives"] == 1   # mauvais comportement
    assert compute_metrics([], [], 100)["precision"] == 1.0


def test_pipeline_survives_empty_scene_and_fps(tmp_path):
    import numpy as np
    from visshop.evaluation import build_demo_pipeline
    sc = build_scenario("normal_browse")
    sc.actors = []
    pipe = build_demo_pipeline(sc, str(tmp_path))
    res = None
    for _ in range(10):
        res = pipe.process_frame(np.zeros((540, 960, 3), np.uint8), 0.0)
    assert res.tracks == [] and res.events == [] and res.frame.shape == (540, 960, 3)


def test_cli_demo_end_to_end(tmp_path, capsys):
    import main as cli
    out = str(tmp_path / "o")
    video = str(tmp_path / "v.mp4")
    assert cli.main(["--demo", "theft_pocket", "--output-dir", out, "--save-video", video]) == 0
    txt = capsys.readouterr().out
    assert "[ALERTE]" in txt and "1 alerte" in txt
    assert os.path.getsize(video) > 1000
    assert cli.main(["--demo", "list"]) == 0
    assert "theft_handbag" in capsys.readouterr().out
