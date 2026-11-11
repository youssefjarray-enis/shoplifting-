"""Rendu 'AI SECURITY MONITOR' : bandeau, etiquettes, squelette, panneau d'incident, synthese."""
import os

import cv2
import numpy as np

from visshop.alert_manager import ActiveAlert
from visshop.evaluation import run_scenario
from visshop.overlay import (PANEL_DURATION_S, Incident, MonitorOverlay, fmt_time,
                             summary_frame, wrist_crop)
from visshop.synthetic import Actor, Key, POCKET_R, build_scenario
from visshop.types import Behavior, Track

BOX = (200, 150, 380, 520)


def _count(img, bgr, tol=40, region=None):
    if region:
        x1, y1, x2, y2 = region
        img = img[y1:y2, x1:x2]
    return int((np.abs(img.astype(int) - np.array(bgr)).sum(axis=2) < tol).sum())


def _inc(ts=3.0):
    return Incident(ts, 1, Behavior.HANDBAG, 0.8, np.full((160, 160, 3), (200, 150, 50), np.uint8), BOX)


def draw(active=False, ts=1.0, incidents=None, poses=None, size=(960, 540)):
    w, h = size
    frame = np.full((h, w, 3), 90, np.uint8)
    act = {1: ActiveAlert(1, Behavior.POCKET, 0.8, BOX)} if active else {}
    stats = dict(total_alerts=int(active), held=int(active), poses=poses or {}, holding=set(),
                 incidents=incidents or [])
    return frame, MonitorOverlay().draw(frame, [Track(1, BOX)], act, stats, ts)


def test_fmt_time():
    assert fmt_time(0) == "00:00" and fmt_time(65.9) == "01:05" and fmt_time(-3) == "00:00"


def test_normal_person_gets_green_id_tab_and_green_header_line():
    frame, out = draw(active=False)
    assert not (out[:60] == frame[:60]).all()                       # bandeau dessine
    assert _count(out, (40, 200, 40)) > 300                          # etiquette "ID: 1" + ligne verte
    assert _count(out, (30, 30, 230), region=BOX) == 0               # pas de rouge
    assert (frame == 90).all()                                       # frame d'entree intacte


def test_suspect_gets_red_box_label_and_red_header_line():
    _, out = draw(active=True)
    assert _count(out, (30, 30, 230)) > 1500
    assert _count(out, (40, 200, 40)) == 0                           # plus de vert


def test_incident_panel_shows_then_disappears():
    inc = [_inc(3.0)]
    _, shown = draw(active=True, ts=3.0 + 1.0, incidents=inc)
    _, gone = draw(active=True, ts=3.0 + PANEL_DURATION_S + 1.0, incidents=inc)
    orange = (40, 160, 240)
    assert _count(shown, orange) > 200 and _count(gone, orange) == 0   # cadre orange de la vignette


def test_skeleton_drawn_only_for_visible_landmarks():
    a = Actor("p", [Key(0, 290, rw=POCKET_R, rvis=0.1)], height=370)
    pose = a.pose_at(0)
    _, out = draw(active=False, poses={1: pose})
    # bras gauche visible -> segments verts ; bras droit cache -> aucun trait vers le poignet droit
    assert _count(out, (40, 200, 40)) > 500


def test_narrow_video_uses_two_line_header():
    frame, out = draw(size=(436, 688))
    hdr = out[:70]
    assert _count(hdr, (255, 255, 255), tol=30) > 100                # titre blanc lisible
    assert _count(out[40:70], (215, 215, 215), tol=60) > 30          # 2e ligne d'infos


def test_wrist_crop_and_summary():
    a = Actor("p", [Key(0, 400, rw=POCKET_R, rvis=0.1)])
    frame = np.random.RandomState(0).randint(0, 255, (540, 960, 3), np.uint8)
    c = wrist_crop(frame, a.pose_at(0), a.bbox_at(0))
    assert c.shape == (160, 160, 3)
    assert wrist_crop(frame, None, (100, 100, 200, 400)).shape == (160, 160, 3)   # sans pose : centre bbox
    assert wrist_crop(None, None, (0, 0, 1, 1)) is None
    assert wrist_crop(np.zeros((5, 5, 3), np.uint8), None, (0, 0, 5, 5)) is None  # frame minuscule
    empty = summary_frame((640, 360), [])
    full = summary_frame((640, 360), [_inc(5.0), _inc(9.0)])
    assert empty.shape == full.shape == (360, 640, 3)
    assert _count(full, (40, 160, 240), region=(0, 0, 640, 360)) > 100   # vignettes orange
    assert not (empty == full).all()


def test_pipeline_collects_incident_pose_and_held_items(tmp_path):
    from visshop.evaluation import build_demo_pipeline
    sc = build_scenario("theft_pocket")
    pipe = build_demo_pipeline(sc, str(tmp_path))
    held_max, saw_panel = 0, False
    for frame, ts in sc.frames():
        res = pipe.process_frame(frame, ts)
        held_max = max(held_max, sum(d.active for d in res.decisions.values()))
        saw_panel |= bool(res.events)
    assert len(pipe.incidents) == 1 and saw_panel and held_max == 1
    inc = pipe.incidents[0]
    assert inc.behavior == Behavior.POCKET and inc.thumb.shape == (160, 160, 3)
    assert res.poses and 1 in res.poses


def test_cli_video_ends_with_summary_card(tmp_path):
    import main as cli
    vid = str(tmp_path / "o.mp4")
    cli.main(["--demo", "theft_pocket", "--output-dir", str(tmp_path / "o"), "--save-video", vid])
    cap = cv2.VideoCapture(vid)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert n >= 315 + 80                                              # 10.5 s + ~3 s de synthese
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 5)
    ok, last = cap.read()
    assert ok and float(last.mean()) < 60                             # ecran sombre de synthese
    classic = str(tmp_path / "c.mp4")
    cli.main(["--demo", "theft_pocket", "--output-dir", str(tmp_path / "c"), "--save-video", classic,
              "--style", "classic"])
    assert int(cv2.VideoCapture(classic).get(cv2.CAP_PROP_FRAME_COUNT)) <= 320   # pas de synthese
