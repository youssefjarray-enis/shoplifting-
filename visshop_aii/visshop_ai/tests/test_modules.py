"""Tests des modules d'E/S : tracker, detecteur, pose, alertes, overlay (avec faux modeles)."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from visshop.alert_manager import AlertManager
from visshop.deepsort_tracker import DeepSortTracker, SimpleIoUTracker, build_tracker
from visshop.false_alert_filter import FilterDecision
from visshop.overlay import Overlay
from visshop.pose_estimator import PoseEstimator
from visshop.types import Behavior, Detection, Track
from visshop.yolo_detector import YoloDetector


def D(x1, y1, x2, y2, label="person"):
    return Detection((x1, y1, x2, y2), 0.9, label)


# ------------------------------ trackers ---------------------------------- #
def test_iou_tracker_keeps_id_while_moving():
    t = SimpleIoUTracker()
    ids = set()
    for i in range(30):
        out = t.update([D(100 + 3 * i, 100, 200 + 3 * i, 400)])
        ids |= {x.track_id for x in out}
    assert ids == {1}


def test_iou_tracker_confirms_after_n_init_and_keeps_through_occlusion():
    t = SimpleIoUTracker(n_init=2, max_age=5)
    assert t.update([D(0, 0, 50, 100)]) == []
    assert len(t.update([D(0, 0, 50, 100)])) == 1
    lost = t.update([])
    assert lost[0].time_since_update == 1               # occulte mais conserve
    back = t.update([D(2, 0, 52, 100)])
    assert back[0].track_id == 1 and back[0].time_since_update == 0


def test_iou_tracker_drops_after_max_age_and_new_id():
    t = SimpleIoUTracker(n_init=1, max_age=2)
    t.update([D(0, 0, 50, 100)])
    for _ in range(4):
        out = t.update([])
    assert out == []
    assert t.update([D(0, 0, 50, 100)])[0].track_id == 2


def test_two_separated_people_get_distinct_ids():
    t = SimpleIoUTracker()
    for _ in range(3):
        out = t.update([D(0, 0, 50, 100), D(300, 0, 350, 100)])
    assert {x.track_id for x in out} == {1, 2}


def test_deepsort_wrapper_with_fake_backend():
    class FakeTrack:
        def __init__(self, i, confirmed=True):
            self.track_id, self._c, self.time_since_update = i, confirmed, 0
        def is_confirmed(self): return self._c
        def to_ltrb(self): return (10, 20, 110, 220)

    class FakeDS:
        def update_tracks(self, raw, frame=None):
            self.raw = raw
            return [FakeTrack(7), FakeTrack(8, confirmed=False)]

    ds = FakeDS()
    out = DeepSortTracker(tracker=ds).update([D(0, 0, 100, 200)], np.zeros((10, 10, 3), np.uint8))
    assert [t.track_id for t in out] == [7] and ds.raw[0][0] == [0, 0, 100, 200]


def test_build_tracker_falls_back_without_deepsort():
    pytest.importorskip("numpy")
    try:
        import deep_sort_realtime  # noqa: F401
        pytest.skip("deep_sort_realtime installe")
    except ImportError:
        assert isinstance(build_tracker(), SimpleIoUTracker)


# ------------------------------ detecteur --------------------------------- #
def test_yolo_parse_results_filters_classes():
    boxes = SimpleNamespace(xyxy=np.array([[0, 0, 10, 20], [5, 5, 9, 9], [1, 1, 2, 2]]),
                            conf=np.array([0.9, 0.8, 0.7]), cls=np.array([0, 26, 2]))
    out = YoloDetector.parse_results([SimpleNamespace(boxes=boxes)])
    assert [d.label for d in out] == ["person", "handbag"]          # classe 2 (voiture) ignoree
    assert YoloDetector.parse_results([]) == []
    assert YoloDetector.parse_results([SimpleNamespace(boxes=None)]) == []


def test_yolo_detect_calls_model_with_device_options():
    calls = {}

    class FakeModel:
        def predict(self, frame, **kw):
            calls.update(kw)
            return [SimpleNamespace(boxes=SimpleNamespace(
                xyxy=np.array([[0, 0, 5, 5]]), conf=np.array([0.5]), cls=np.array([24])))]

    from visshop.model_selector import DeviceInfo, select_model
    det = YoloDetector(select_model(DeviceInfo(False)), conf=0.3, model=FakeModel())
    out = det.detect(np.zeros((50, 50, 3), np.uint8))
    assert out[0].label == "backpack"
    assert calls["classes"] == [0, 24, 26] and calls["device"] == "cpu" and calls["conf"] == 0.3


# ------------------------------ pose -------------------------------------- #
class FakeLM:
    def __init__(self, x, y, vis=0.9):
        self.x, self.y, self.z, self.visibility = x, y, 0.0, vis


def test_pose_estimator_converts_roi_to_frame_coordinates():
    class FakeHolistic:
        def process(self, rgb):
            assert rgb.shape[2] == 3
            body = SimpleNamespace(landmark=[FakeLM(0.5, 0.5) for _ in range(33)])
            hand = SimpleNamespace(landmark=[FakeLM(0.0, 1.0) for _ in range(21)])
            return SimpleNamespace(pose_landmarks=body, left_hand_landmarks=hand,
                                   right_hand_landmarks=None)

    frame = np.zeros((600, 800, 3), np.uint8)
    pe = PoseEstimator(holistic=FakeHolistic(), padding=0.0)
    pose = pe.estimate(frame, (100, 100, 300, 500))
    assert pose.body.shape == (33, 4)
    assert pose.point(0) == (200.0, 300.0)                      # centre de la ROI
    assert pose.left_hand.shape == (21, 3) and pose.right_hand is None
    assert tuple(pose.left_hand[0, :2]) == (100.0, 500.0)


def test_pose_estimator_handles_no_person_and_tiny_roi():
    class Empty:
        def process(self, rgb):
            return SimpleNamespace(pose_landmarks=None, left_hand_landmarks=None, right_hand_landmarks=None)

    frame = np.zeros((300, 300, 3), np.uint8)
    pe = PoseEstimator(holistic=Empty())
    assert pe.estimate(frame, (10, 10, 200, 280)) is None
    assert pe.estimate(frame, (10, 10, 20, 20)) is None        # ROI trop petite


# ------------------------------ alertes + overlay ------------------------- #
def decision(tid=1, active=True, new=True, beh=Behavior.POCKET, score=0.81):
    return FilterDecision(tid, active, beh if active else None, score, new)


def test_alert_manager_logs_json_and_screenshot(tmp_path):
    am = AlertManager(str(tmp_path), "camX")
    frame = np.zeros((200, 200, 3), np.uint8)
    evs = am.handle(frame, {1: decision()}, {1: (20, 20, 120, 180)}, 12.5)
    assert len(evs) == 1 and am.total_events == 1
    rows = AlertManager.read_log(am.log_path)
    assert rows[0]["track_id"] == 1 and rows[0]["behavior"] == "vol_en_poche"
    assert rows[0]["camera_id"] == "camX" and rows[0]["frame_ts"] == 12.5
    assert "T" in rows[0]["timestamp"]                           # ISO 8601 horodate
    assert os.path.exists(rows[0]["screenshot"])
    json.loads(open(am.log_path).readline())                     # JSON Lines valide
    assert 1 in am.active_alerts
    am.handle(frame, {1: decision(new=False)}, {1: (20, 20, 120, 180)}, 12.6)
    assert am.total_events == 1                                   # pas de re-journalisation


def test_alert_manager_without_screenshots_and_missing_track(tmp_path):
    am = AlertManager(str(tmp_path), save_screenshots=False)
    frame = np.zeros((50, 50, 3), np.uint8)
    assert am.handle(frame, {9: decision(9)}, {}, 0.0) == []     # track absente -> ignoree
    ev = am.handle(frame, {1: decision()}, {1: (0, 0, 10, 10)}, 0.0)[0]
    assert ev.screenshot is None


def _red_pixels(img, box):
    x1, y1, x2, y2 = box
    reg = img[y1 - 8:y2 + 8, x1 - 8:x2 + 8].astype(int)
    return int(((reg[..., 2] > 200) & (reg[..., 1] < 60) & (reg[..., 0] < 60)).sum())


def test_overlay_flashes_red_frame(tmp_path):
    from visshop.alert_manager import ActiveAlert
    ov = Overlay(flash_hz=2.0)
    frame = np.full((400, 600, 3), 90, np.uint8)
    box = (200, 100, 350, 380)
    tracks = [Track(1, box)]
    active = {1: ActiveAlert(1, Behavior.POCKET, 0.8, box)}
    stats = dict(camera_id="c", fps=12.0, persons=1, total_alerts=1, recent=["Vol en poche ID1"])
    on = ov.draw(frame, tracks, active, stats, ts=0.0)
    off = ov.draw(frame, tracks, active, stats, ts=0.25)
    calm = ov.draw(frame, tracks, {}, stats, ts=0.0)
    assert ov.flash_on(0.0) and not ov.flash_on(0.25)
    assert _red_pixels(on, box) > 500
    assert _red_pixels(off, box) == 0 and _red_pixels(calm, box) == 0
    assert (frame == 90).all()                                    # frame d'entree non modifiee
