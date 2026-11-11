import numpy as np
import pytest

from visshop.action_classifier import ActionClassifier
from visshop.config import BEHAVIOR_THRESHOLDS
from visshop.features import BagMemory, associate_bags, build_geometry, hand_grip
from visshop.synthetic import (BACKPACK_BOX, BACKPACK_WRIST_R, BELLY_R, HANDBAG_BOX,
                               HANDBAG_WRIST_R, HANG_R, POCKET_R, REACH_R, Actor, BagSpec,
                               Key, make_hand)
from visshop.types import Behavior, Detection, Track


def test_hand_grip_open_vs_closed():
    assert hand_grip(make_hand(100, 100, 20, 0.0)) < 0.15
    assert hand_grip(make_hand(100, 100, 20, 1.0)) > 0.85
    assert hand_grip(make_hand(100, 100, 20, 0.5)) < hand_grip(make_hand(100, 100, 20, 0.9))
    assert hand_grip(None) is None


def test_geometry_from_pose_and_fallback():
    a = Actor("p", [Key(0, 300)])
    pose, bbox = a.pose_at(0), a.bbox_at(0)
    g = build_geometry(pose, bbox)
    assert abs(g.torso_len - 0.32 * 320) < 1.0
    pose.body[:, 3] = 0.0   # aucun landmark fiable -> proportions de la bbox
    g2 = build_geometry(pose, bbox)
    assert abs(g2.torso_len - g.torso_len) < 0.08 * g.torso_len


def _run(keys, bags=None, shelf=1.0, times=None):
    """Rejoue des keyframes via le classifieur ; retourne le dernier jeu de scores."""
    actor = Actor("p", keys, bags=bags or [])
    clf, scores = ActionClassifier(), None
    for t in times:
        dets = {d.label: d.bbox for d in actor.bag_dets_at(t)}
        scores = clf.classify(1, actor.pose_at(t), actor.bbox_at(t), t, shelf, dets)
    return scores


def _theft(conceal, bags=None):
    keys = [Key(0, 600, rw=REACH_R, rgrip=0.9), Key(1.0, 600, rw=REACH_R, rgrip=0.9),
            Key(1.5, 600, rw=conceal, rvis=0.1, rgrip=0.9), Key(4, 600, rw=conceal, rvis=0.1, rgrip=0.9)]
    return _run(keys, bags, times=np.arange(0, 2.5, 0.1))


@pytest.mark.parametrize("behavior,conceal,bags", [
    (Behavior.POCKET, POCKET_R, None),
    (Behavior.UNDER_CLOTHING, BELLY_R, None),
    (Behavior.BACKPACK, BACKPACK_WRIST_R, [BagSpec("backpack", BACKPACK_BOX)]),
    (Behavior.HANDBAG, HANDBAG_WRIST_R, [BagSpec("handbag", HANDBAG_BOX)]),
])
def test_concealment_scores_high_for_right_behavior(behavior, conceal, bags):
    s = _theft(conceal, bags)
    assert s[behavior] >= BEHAVIOR_THRESHOLDS[behavior]
    assert s[behavior] == max(s.values())


def test_relaxed_arms_score_low():
    s = _run([Key(0, 600)], times=np.arange(0, 3, 0.1), shelf=0.0)
    assert max(s.values()) < 0.3


def test_hand_visible_at_shelf_scores_low():
    keys = [Key(0, 600), Key(1, 600, rw=REACH_R, rgrip=0.9), Key(3, 600, rw=REACH_R, rgrip=0.9)]
    s = _run(keys, times=np.arange(0, 3, 0.1))
    assert max(s.values()) < 0.6


def test_bag_behaviors_need_a_bag():
    s = _theft(HANDBAG_WRIST_R, bags=None)
    assert s[Behavior.HANDBAG] == 0.0 and s[Behavior.BACKPACK] == 0.0


def test_missing_pose_returns_zero():
    assert set(ActionClassifier().classify(1, None, (0, 0, 10, 10), 0.0).values()) == {0.0}


def test_associate_bags():
    tracks = [Track(1, (100, 100, 200, 400)), Track(2, (600, 100, 700, 400))]
    bags = [Detection((190, 250, 230, 320), 0.8, "handbag"),
            Detection((580, 150, 620, 250), 0.7, "backpack"),
            Detection((900, 900, 950, 950), 0.9, "handbag")]   # sans proprietaire
    out = associate_bags(tracks, bags)
    assert out[1]["handbag"] == bags[0].bbox and out[1]["backpack"] is None
    assert out[2]["backpack"] == bags[1].bbox


# ---------------- gros plans / vraies videos ---------------------------------- #
def _closeup_pose(width=250.0):
    """Plan serre : seules les epaules sont dans le cadre ; hanches 'inventees' hors bbox."""
    from visshop.types import LM, PoseResult
    body = np.zeros((33, 4), np.float32)
    body[:, 3] = 0.99
    body[LM.L_SHOULDER] = (200, 250, 0, 1.0)
    body[LM.R_SHOULDER] = (200 + width, 250, 0, 1.0)
    body[LM.L_HIP] = (210, 900, 0, 0.99)       # hors cadre (bbox s'arrete a 688)
    body[LM.R_HIP] = (200 + width - 10, 900, 0, 0.99)
    return PoseResult(body)


def test_closeup_geometry_ignores_hallucinated_hips():
    g = build_geometry(_closeup_pose(250.0), (0, 0, 436, 688))
    assert abs(g.torso_len - 1.3 * 250) < 1.0           # deduit de la largeur des epaules
    assert g.pockets["L"][1] < 688 * 1.5


def test_score_without_detected_hands_is_renormalised():
    """Sans mains detectees (gros plan), le critere 'prehension' est retire : un vol reste
    detectable et la posture normale reste basse."""
    steal = [Key(0, 600, rw=REACH_R, rvis=0.1), Key(3, 600, rw=HANDBAG_WRIST_R, rvis=0.1)]
    actor = Actor("p", steal, bags=[BagSpec("handbag", HANDBAG_BOX)])
    clf, sc = ActionClassifier(), None
    for t in np.arange(0, 3, 0.1):
        pose = actor.pose_at(t)
        pose.left_hand = pose.right_hand = None               # aucune main detectee
        dets = {d.label: d.bbox for d in actor.bag_dets_at(t)}
        sc = clf.classify(1, pose, actor.bbox_at(t), t, 1.0, dets)
    assert sc[Behavior.HANDBAG] >= BEHAVIOR_THRESHOLDS[Behavior.HANDBAG]
    calm = Actor("q", [Key(0, 600)], bags=[BagSpec("handbag", HANDBAG_BOX)])
    clf2 = ActionClassifier()
    for t in np.arange(0, 3, 0.1):
        pose = calm.pose_at(t)
        pose.left_hand = pose.right_hand = None
        dets = {d.label: d.bbox for d in calm.bag_dets_at(t)}
        sc2 = clf2.classify(1, pose, calm.bbox_at(t), t, 1.0, dets)
    assert max(sc2.values()) < 0.6


def test_pose_loss_is_bridged_then_drops():
    actor = Actor("p", [Key(0, 600, rw=POCKET_R, rvis=0.1, rgrip=0.9)])
    clf = ActionClassifier()
    ref = clf.classify(1, actor.pose_at(0), actor.bbox_at(0), 0.0, 1.0)
    held = clf.classify(1, None, actor.bbox_at(0), 0.1)
    assert 0 < held[Behavior.POCKET] <= ref[Behavior.POCKET] * 0.9 + 1e-9
    for _ in range(10):
        out = clf.classify(1, None, actor.bbox_at(0), 0.2)
    assert set(out.values()) == {0.0}


def test_bag_memory_bridges_missed_detections():
    m = BagMemory(memory_s=1.5)
    box = (10.0, 10.0, 50.0, 50.0)
    assert m.update({1: {"handbag": box, "backpack": None}}, 0.0)[1]["handbag"] == box
    assert m.update({1: {"handbag": None, "backpack": None}}, 1.0)[1]["handbag"] == box   # memorise
    assert m.update({1: {"handbag": None, "backpack": None}}, 3.0)[1]["handbag"] is None  # expire
    m.update({1: {"handbag": box, "backpack": None}}, 4.0)
    m.remove_track(1)
    assert m.update({1: {"handbag": None, "backpack": None}}, 4.1)[1]["handbag"] is None
