import json

from visshop.config import BEHAVIOR_THRESHOLDS, Config, ZoneConfig, load_config
from visshop.types import (Behavior, bbox_center, distance_point_bbox, iou,
                           point_in_polygon)


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-6


def test_point_in_polygon():
    sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert point_in_polygon((5, 5), sq)
    assert not point_in_polygon((15, 5), sq)
    assert not point_in_polygon((5, 5), [[0, 0], [1, 1]])


def test_distance_point_bbox():
    assert distance_point_bbox((5, 5), (0, 0, 10, 10)) == 0.0
    assert distance_point_bbox((13, 14), (0, 0, 10, 10)) == 5.0
    assert bbox_center((0, 0, 10, 20)) == (5, 10)


def test_default_thresholds_match_spec():
    assert BEHAVIOR_THRESHOLDS[Behavior.POCKET] == 0.72
    assert BEHAVIOR_THRESHOLDS[Behavior.UNDER_CLOTHING] == 0.78
    assert BEHAVIOR_THRESHOLDS[Behavior.BACKPACK] == 0.70
    assert BEHAVIOR_THRESHOLDS[Behavior.HANDBAG] == 0.75
    f = Config().filter
    assert (f.smoothing_window, f.trigger_threshold, f.cancel_threshold, f.cooldown_s) == (20, 0.68, 0.35, 3.0)


def test_zone_scaling():
    z = ZoneConfig(shelf=[[[0.5, 0.5], [1, 0.5], [1, 1]]], normalized=True).scaled(200, 100)
    assert z.shelf[0][0] == [100.0, 50.0] and not z.normalized


def test_load_config_json(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"thresholds": {"vol_en_poche": 0.9},
                             "filter": {"cooldown_s": 5.0},
                             "staff": {"hsv_ranges": [[[1, 2, 3], [4, 5, 6]]]}}))
    cfg = load_config(str(p))
    assert cfg.thresholds[Behavior.POCKET] == 0.9
    assert cfg.filter.cooldown_s == 5.0
    assert cfg.staff.hsv_ranges == [((1, 2, 3), (4, 5, 6))]
    assert load_config(None).filter.cooldown_s == 3.0


def test_example_config_loads():
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.example.json")
    cfg = load_config(path)
    assert cfg.zones.normalized and cfg.zones.shelf
