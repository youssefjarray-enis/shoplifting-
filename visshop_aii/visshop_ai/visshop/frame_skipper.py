"""Frame Skipping adaptatif : MediaPipe/classification tournent 1 frame sur N."""
from __future__ import annotations


class AdaptiveFrameSkipper:
    """Ajuste N pour tenir le temps reel.

    Si le temps moyen par frame depasse le budget (1/fps source) -> N augmente ;
    s'il y a beaucoup de marge -> N diminue (plus de reactivite).
    YOLO + DeepSORT tournent a chaque frame ; seule l'analyse de pose est espacee.
    """

    def __init__(self, base_n: int = 3, min_n: int = 2, max_n: int = 10,
                 source_fps: float = 30.0, adaptive: bool = True, adjust_every: int = 15):
        self.n = max(1, int(base_n))
        self.min_n, self.max_n = min_n, max_n
        self.budget = 1.0 / max(source_fps, 1.0)
        self.adaptive = adaptive
        self.adjust_every = adjust_every
        self._ema = None
        self._count = 0

    def should_analyze(self, frame_idx: int) -> bool:
        return frame_idx % self.n == 0

    def report(self, frame_time_s: float) -> None:
        self._ema = frame_time_s if self._ema is None else 0.9 * self._ema + 0.1 * frame_time_s
        self._count += 1
        if not self.adaptive or self._count % self.adjust_every:
            return
        if self._ema > self.budget * 1.05:
            self.n = min(self.max_n, self.n + 1)
        elif self._ema < self.budget * 0.55:
            self.n = max(self.min_n, self.n - 1)
