import numpy as np

from visshop.config import BEHAVIOR_THRESHOLDS, FilterConfig, StaffConfig, ZoneConfig
from visshop.false_alert_filter import (ContextAnalyzer, ContextInfo, FalseAlertFilter,
                                        StaffDetector)
from visshop.types import Behavior

P = Behavior.POCKET


def mk():
    return FalseAlertFilter(FilterConfig(), dict(BEHAVIOR_THRESHOLDS))


def scores(v, b=P):
    return {x: (v if x == b else 0.0) for x in Behavior}


def feed(f, vals, t0=0.0, dt=0.1, ctx=None, tid=1, b=P):
    out = []
    for i, v in enumerate(vals):
        out.append(f.update(tid, scores(v, b), t0 + i * dt, ctx or ContextInfo()))
    return out


def test_sustained_high_score_triggers_once():
    d = feed(mk(), [0.0] * 5 + [0.95] * 30)
    assert sum(x.new_event for x in d) == 1
    assert d[-1].active and d[-1].behavior == P


def test_quick_spike_does_not_trigger():           # geste rapide inoffensif
    d = feed(mk(), [0.0] * 10 + [0.95] * 3 + [0.0] * 10)
    assert not any(x.active for x in d)


def test_min_samples_required():
    d = feed(mk(), [0.95] * 5)
    assert not any(x.active for x in d)


def test_smoothing_is_weighted_towards_recent():
    f = FalseAlertFilter(FilterConfig(), dict(BEHAVIOR_THRESHOLDS))
    a = feed(f, [0.0] * 10 + [1.0] * 10)[-1].smoothed[P]
    b = feed(FalseAlertFilter(FilterConfig(), dict(BEHAVIOR_THRESHOLDS)), [1.0] * 10 + [0.0] * 10)[-1].smoothed[P]
    assert a > 0.5 > b


def test_hysteresis_holds_then_cancels():
    f = mk()
    feed(f, [0.95] * 25)                                  # declenche
    mid = feed(f, [0.5] * 3, t0=2.5)                      # baisse partielle : reste actif
    assert mid[-1].active
    end = feed(f, [0.0] * 30, t0=2.8)                     # chute : annule (< 0.35)
    assert not end[-1].active


def test_raw_score_must_reach_behavior_threshold():
    # lisse >= 0.68 mais score brut < 0.72 (seuil poche) -> pas d'alerte
    d = feed(mk(), [0.70] * 40)
    assert not any(x.active for x in d)


def test_cooldown_blocks_second_log_within_3s():
    f = mk()
    d1 = feed(f, [0.95] * 20)
    assert sum(x.new_event for x in d1) == 1
    feed(f, [0.0] * 30, t0=2.0)                           # annulation
    d2 = feed(f, [0.95] * 25, t0=5.0 - 2.0 + 2.0 + 0.0)   # re-declenche vers t~5-6 (> 3s apres)
    assert any(x.new_event for x in d2)

    g = mk()
    feed(g, [0.95] * 20)
    feed(g, [0.0] * 8, t0=2.0)
    d3 = feed(g, [0.95] * 25, t0=2.8)
    # re-declenchement < 3 s apres la 1re alerte : alerte visuelle OK mais pas de nouvelle ligne de log
    assert not any(x.new_event for x in d3)


def test_context_rules_suppress():
    for ctx, reason in [(ContextInfo(is_staff=True), "staff"),
                        (ContextInfo(in_fitting_room=True), "fitting_room"),
                        (ContextInfo(shelf_visited=False), "no_shelf_visit")]:
        d = feed(mk(), [0.99] * 40, ctx=ctx)
        assert not any(x.active for x in d) and d[-1].suppressed_reason == reason


def test_alert_cleared_when_entering_fitting_room():
    f = mk()
    d = feed(f, [0.95] * 20)
    assert d[-1].active
    d = feed(f, [0.95] * 2, t0=2.0, ctx=ContextInfo(in_fitting_room=True))
    assert not d[-1].active


def test_tracks_are_independent_and_removable():
    f = mk()
    feed(f, [0.95] * 20, tid=1)
    assert not feed(f, [0.0] * 20, tid=2)[-1].active
    f.remove_track(1)
    assert not feed(f, [0.95] * 3, tid=1)[-1].active    # etat remis a zero


def test_best_behavior_is_selected():
    f = mk()
    out = None
    for i in range(30):
        s = {Behavior.POCKET: 0.75, Behavior.UNDER_CLOTHING: 0.95, Behavior.BACKPACK: 0.0, Behavior.HANDBAG: 0.0}
        out = f.update(1, s, i * 0.1, ContextInfo())
    assert out.behavior == Behavior.UNDER_CLOTHING


# ---- contexte : zones rayon / cabine / personnel ----------------------------- #
ZONES = ZoneConfig(shelf=[[[500, 400], [900, 400], [900, 540], [500, 540]]],
                   fitting_room=[[[0, 400], [200, 400], [200, 540], [0, 540]]])


def box_at(cx):
    return (cx - 50, 200, cx + 50, 500)


def test_shelf_dwell_required():
    c = ContextAnalyzer(ZONES, FilterConfig())
    assert not c.update(1, box_at(600), 0.0).shelf_visited      # vient d'entrer
    assert not c.update(1, box_at(600), 0.5).shelf_visited
    ctx = c.update(1, box_at(600), 1.2)
    assert ctx.shelf_visited and ctx.in_shelf and ctx.shelf_recent == 1.0


def test_shelf_memory_decays_and_expires():
    c = ContextAnalyzer(ZONES, FilterConfig())
    for t in np.arange(0, 2, 0.2):
        c.update(1, box_at(600), t)
    mid = c.update(1, box_at(350), 5.0)
    assert mid.shelf_visited and 0 < mid.shelf_recent < 1
    assert not c.update(1, box_at(350), 20.0).shelf_visited


def test_fitting_room_and_no_zones():
    c = ContextAnalyzer(ZONES, FilterConfig())
    assert c.update(1, box_at(100), 0.0).in_fitting_room
    n = ContextAnalyzer(ZoneConfig(), FilterConfig()).update(1, box_at(100), 0.0)
    assert n.shelf_visited and n.shelf_recent == 1.0 and not n.in_fitting_room


def test_staff_detection_by_uniform_colour():
    det = StaffDetector(StaffConfig(hsv_ranges=[((100, 100, 50), (130, 255, 255))]))
    img = np.full((600, 400, 3), 70, np.uint8)
    img[250:350, 150:250] = (200, 100, 30)      # BGR bleu
    assert det.vote(img, (100, 200, 300, 500)) is True
    assert det.vote(np.full((600, 400, 3), 70, np.uint8), (100, 200, 300, 500)) is False
    assert StaffDetector(StaffConfig()).vote(img, (100, 200, 300, 500)) is None
    c = ContextAnalyzer(ZoneConfig(), FilterConfig())
    ctx = None
    for i in range(4):
        ctx = c.update(1, (100, 200, 300, 500), i, True)
    assert ctx.is_staff
