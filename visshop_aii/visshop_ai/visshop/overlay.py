"""Overlay video : cadre rouge clignotant sur la personne suspecte + dashboard."""
from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

from .alert_manager import ActiveAlert
from .types import Track

RED, GREEN, WHITE = (0, 0, 255), (0, 200, 0), (255, 255, 255)


class Overlay:
    def __init__(self, flash_hz: float = 2.0):
        self.flash_hz = flash_hz

    def flash_on(self, ts: float) -> bool:
        return int(ts * self.flash_hz * 2) % 2 == 0

    def draw(self, frame: np.ndarray, tracks: List[Track],
             active: Dict[int, ActiveAlert], stats: dict, ts: float) -> np.ndarray:
        img = frame.copy()
        on = self.flash_on(ts)
        for t in tracks:
            x1, y1, x2, y2 = (int(v) for v in t.bbox)
            alert = active.get(t.track_id)
            if alert is None:
                cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)
                cv2.putText(img, f"ID{t.track_id}", (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)
            elif on:
                cv2.rectangle(img, (x1, y1), (x2, y2), RED, 5)
                cv2.rectangle(img, (x1, max(0, y1 - 24)), (x2, y1), RED, -1)
                cv2.putText(img, f"ALERTE {alert.behavior.label_fr} {alert.score:.2f}",
                            (x1 + 4, max(14, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), WHITE, 1)
        self._dashboard(img, stats)
        return img

    @staticmethod
    def _dashboard(img: np.ndarray, stats: dict) -> None:
        lines = [f"VisShop AI | {stats.get('camera_id', 'cam')}",
                 f"FPS: {stats.get('fps', 0):.1f}   Personnes: {stats.get('persons', 0)}",
                 f"Alertes (session): {stats.get('total_alerts', 0)}"]
        lines += [f"  {e}" for e in stats.get("recent", [])[-3:]]
        h = 8 + 18 * len(lines)
        panel = img[0:h, 0:360].copy()
        cv2.rectangle(img, (0, 0), (360, h), (20, 20, 20), -1)
        cv2.addWeighted(panel, 0.35, img[0:h, 0:360], 0.65, 0, img[0:h, 0:360])
        for i, line in enumerate(lines):
            cv2.putText(img, line, (6, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, WHITE, 1, cv2.LINE_AA)


# =========================================================================== #
# Style "AI SECURITY MONITOR" : bandeau, etiquettes ID, squelette, panneau d'incident
# =========================================================================== #
from dataclasses import dataclass
from typing import Optional

from .types import Behavior, LM, PoseResult

GREEN_D, RED_D, YELLOW, ORANGE = (40, 200, 40), (30, 30, 230), (0, 230, 255), (40, 160, 240)
FONT = cv2.FONT_HERSHEY_DUPLEX
SKELETON = [(LM.L_SHOULDER, LM.L_ELBOW), (LM.L_ELBOW, LM.L_WRIST), (LM.R_SHOULDER, LM.R_ELBOW),
            (LM.R_ELBOW, LM.R_WRIST), (LM.L_SHOULDER, LM.R_SHOULDER)]
PANEL_DURATION_S = 6.0


@dataclass
class Incident:
    """Evenement journalise + vignette de preuve (zone de la main suspecte)."""
    ts: float
    track_id: int
    behavior: Behavior
    score: float
    thumb: Optional[np.ndarray] = None   # BGR 160x160
    bbox: Optional[tuple] = None


def fmt_time(ts: float) -> str:
    s = int(max(0, ts))
    return f"{s // 60:02d}:{s % 60:02d}"


def _txt(img, text, org, scale, color, thick=1, font=FONT):
    cv2.putText(img, text, org, font, scale, color, thick, cv2.LINE_AA)


def _tab(img, text, x, y, scale, bg, fg=(255, 255, 255)):
    """Etiquette pleine (ex. 'ID: 1') dont le bord bas est en y."""
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
    pad = max(4, int(th * 0.45))
    x1, y1, x2, y2 = x, y - th - 2 * pad, x + tw + 2 * pad, y
    cv2.rectangle(img, (x1, y1), (x2, y2), bg, -1)
    _txt(img, text, (x1 + pad, y2 - pad), scale, fg, 2)


def wrist_crop(frame: np.ndarray, pose: Optional[PoseResult], bbox, size: int = 160) -> Optional[np.ndarray]:
    """Vignette de preuve centree sur le poignet le moins visible (main cachee)."""
    if frame is None:
        return None
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    half = max(40, int(0.22 * (y2 - y1)))
    if pose is not None:
        side = min((LM.L_WRIST, LM.R_WRIST), key=pose.visibility)
        cx, cy = pose.point(side)
    else:
        cx, cy = (x1 + x2) / 2, y1 + 0.6 * (y2 - y1)
    cx, cy = min(max(cx, 0), W - 1), min(max(cy, 0), H - 1)
    a, b = int(max(0, cx - half)), int(max(0, cy - half))
    c, d = int(min(W, cx + half)), int(min(H, cy + half))
    if c - a < 8 or d - b < 8:
        return None
    return cv2.resize(frame[b:d, a:c], (size, size))


class MonitorOverlay(Overlay):
    """Rendu type 'AI SECURITY MONITOR' (stats.poses / holding / incidents / total_alerts)."""

    def __init__(self, flash_hz: float = 2.0):
        super().__init__(flash_hz)

    def draw(self, frame, tracks, active, stats, ts):
        img = frame.copy()
        H, W = img.shape[:2]
        alert_on = bool(active)
        hh = self._header(img, ts, stats, alert_on)
        poses = stats.get("poses", {})
        holding = set(stats.get("holding", ()))
        blink = self.flash_on(ts)
        for t in tracks:
            x1, y1, x2, y2 = (int(v) for v in t.bbox)
            suspect = t.track_id in active
            col = RED_D if suspect else GREEN_D
            pose = poses.get(t.track_id)
            if pose is not None:
                self._skeleton(img, pose, col, W, H, joints=suspect or t.track_id in holding)
            ty = max(hh + 28, y1)
            tx = min(max(x1 + (x2 - x1) // 3, 6), max(6, W - 150))
            scale = max(0.5, min(W / 1128 * 0.9, 0.9))
            if suspect:
                cv2.rectangle(img, (x1, y1), (x2, y2), RED_D, 4 if blink else 2)
                _tab(img, f"SUSPECT {t.track_id}", tx, ty, scale, RED_D)
                self._item_tag(img, pose, t.bbox, scale)
            else:
                _tab(img, f"ID: {t.track_id}", tx, ty, scale, GREEN_D)
        inc = stats.get("incidents") or []
        if inc and ts - inc[-1].ts <= PANEL_DURATION_S:
            self._panel(img, inc[-1])
        return img

    # -- elements ---------------------------------------------------------- #
    def _header(self, img, ts, stats, alert_on) -> int:
        H, W = img.shape[:2]
        held, alerts = stats.get("held", 0), stats.get("total_alerts", 0)
        right = f"{fmt_time(ts)} | Held Items: {held} | Alerts: {alerts}"
        title = "AI SECURITY MONITOR"

        def width(text, fs, th):
            return cv2.getTextSize(text, FONT, fs, th)[0][0]

        fs = min(0.95, W / 1128)
        one_line = W >= 720
        if one_line:
            while fs > 0.35 and width(title, fs, 2) + width(right, fs * 0.55, 1) + 40 > W:
                fs -= 0.05
            hh = int(min(56, max(36, 0.065 * H)))
        else:                      # video etroite : titre puis infos sur 2 lignes
            fs = 0.8
            while fs > 0.4 and width(title, fs, 2) > W - 16:
                fs -= 0.05
            hh = int(min(90, max(60, 0.115 * H)))
        over = img[0:hh].copy()
        cv2.rectangle(img, (0, 0), (W, hh), (25, 25, 25), -1)
        cv2.addWeighted(over, 0.45, img[0:hh], 0.55, 0, img[0:hh])
        if one_line:
            base = int(hh * 0.68)
            _txt(img, title, (8, base), fs, (255, 255, 255), 2)
            _txt(img, right, (8 + width(title, fs, 2) + 18, base), fs * 0.55, (210, 210, 210), 1)
        else:
            _txt(img, title, (8, int(hh * 0.42)), fs, (255, 255, 255), 2)
            fs2 = fs * 0.62
            while fs2 > 0.3 and width(right, fs2, 1) > W - 16:
                fs2 -= 0.03
            _txt(img, right, (8, int(hh * 0.85)), fs2, (215, 215, 215), 1)
        cv2.line(img, (0, hh), (W, hh), RED_D if alert_on else GREEN_D, 4)
        return hh

    @staticmethod
    def _skeleton(img, pose, col, W, H, joints: bool):
        def ok(i):
            x, y = pose.point(i)
            return pose.visibility(i) >= 0.4 and 0 <= x < W and 0 <= y < H
        for a, b in SKELETON:
            if ok(a) and ok(b):
                cv2.line(img, tuple(int(v) for v in pose.point(a)),
                         tuple(int(v) for v in pose.point(b)), col, 3, cv2.LINE_AA)
        if joints:
            for i in (LM.L_WRIST, LM.R_WRIST):
                if ok(i):
                    cv2.circle(img, tuple(int(v) for v in pose.point(i)), 7, YELLOW, -1, cv2.LINE_AA)

    @staticmethod
    def _item_tag(img, pose, bbox, scale):
        """Cadre blanc 'Item ?' pres de la main suspecte (la moins visible)."""
        H, W = img.shape[:2]
        x1, y1, x2, y2 = bbox
        if pose is None:
            return
        side = min((LM.L_WRIST, LM.R_WRIST), key=pose.visibility)
        cx, cy = pose.point(side)
        half = int(max(18, 0.07 * (y2 - y1)))
        cx, cy = int(min(max(cx, half), W - half)), int(min(max(cy, half + 20), H - half))
        cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), (255, 255, 255), 2)
        _txt(img, "Item ?", (cx - half, cy - half - 6), scale * 0.5, (255, 255, 255), 1)

    @staticmethod
    def _panel(img, inc: Incident):
        H, W = img.shape[:2]
        pw = int(min(W - 20, max(0.42 * W, 250)))
        ph = int(pw * 0.98)
        x0, y0 = W - pw - 10, int(min(max(H * 0.25, 70), max(70, H - ph - 10)))
        reg = img[y0:y0 + ph, x0:x0 + pw].copy()
        cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (15, 15, 15), -1)
        cv2.addWeighted(reg, 0.30, img[y0:y0 + ph, x0:x0 + pw], 0.70, 0, img[y0:y0 + ph, x0:x0 + pw])
        cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), RED_D, 2)
        fs = pw / 520.0
        pad = int(pw * 0.06)
        y = y0 + int(pw * 0.11)
        _txt(img, "INCIDENT RECORD", (x0 + pad, y), fs * 0.95, RED_D, 2)
        y += int(pw * 0.095)
        _txt(img, f"Time: {fmt_time(inc.ts)}", (x0 + pad, y), fs * 0.72, (235, 235, 235), 1)
        y += int(pw * 0.085)
        _txt(img, f"Suspect ID: {inc.track_id}", (x0 + pad, y), fs * 0.72, (235, 235, 235), 1)
        y += int(pw * 0.085)
        _txt(img, inc.behavior.label_fr.upper(), (x0 + pad, y), fs * 0.62, (255, 255, 255), 1)
        y += int(pw * 0.03)
        ts_ = int(min(pw - 2 * pad, y0 + ph - y - 12))
        if inc.thumb is not None and ts_ > 30:
            th = cv2.resize(inc.thumb, (ts_, ts_))
            xx = x0 + (pw - ts_) // 2
            img[y:y + ts_, xx:xx + ts_] = th
            cv2.rectangle(img, (xx, y), (xx + ts_, y + ts_), ORANGE, 2)


def summary_frame(size, incidents, camera_id: str = "") -> np.ndarray:
    """Ecran final 'INCIDENT REPORT SUMMARY'."""
    W, H = size
    img = np.full((H, W, 3), (12, 12, 12), np.uint8)
    fs = max(0.5, min(1.0, W / 900))
    title = "INCIDENT REPORT SUMMARY"
    (tw, th), _ = cv2.getTextSize(title, FONT, fs, 2)
    _txt(img, title, ((W - tw) // 2, int(H * 0.12)), fs, (255, 255, 255), 2)
    cv2.line(img, (int(W * 0.1), int(H * 0.16)), (int(W * 0.9), int(H * 0.16)), RED_D, 2)
    if not incidents:
        msg = "No suspicious behaviour detected"
        (mw, _), _ = cv2.getTextSize(msg, FONT, fs * 0.8, 1)
        _txt(img, msg, ((W - mw) // 2, H // 2), fs * 0.8, GREEN_D, 1)
        return img
    row_h = int(min(130, (H * 0.78) / max(1, min(len(incidents), 5))))
    y = int(H * 0.22)
    for n, inc in enumerate(incidents[:5], 1):
        t = row_h - 14
        if inc.thumb is not None and t > 30:
            img[y:y + t, int(W * 0.08):int(W * 0.08) + t] = cv2.resize(inc.thumb, (t, t))
            cv2.rectangle(img, (int(W * 0.08), y), (int(W * 0.08) + t, y + t), ORANGE, 2)
        x = int(W * 0.08) + t + 20
        _txt(img, f"Incident #{n}  -  {fmt_time(inc.ts)}", (x, y + int(t * 0.30)), fs * 0.7, RED_D, 2)
        _txt(img, f"Suspect ID: {inc.track_id}   Score: {inc.score:.2f}", (x, y + int(t * 0.58)), fs * 0.6, (230, 230, 230), 1)
        _txt(img, inc.behavior.label_fr.upper(), (x, y + int(t * 0.86)), fs * 0.6, (255, 255, 255), 1)
        y += row_h
    if len(incidents) > 5:
        _txt(img, f"+ {len(incidents) - 5} more", (int(W * 0.08), H - 20), fs * 0.6, (200, 200, 200), 1)
    return img
