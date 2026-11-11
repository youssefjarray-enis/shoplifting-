"""Etape 1 : acquisition video (RTSP / USB / camera IP / fichier) via OpenCV."""
from __future__ import annotations

import logging
import os
import time
from typing import Iterator, Tuple, Union

import cv2
import numpy as np

log = logging.getLogger(__name__)


class VideoSource:
    """Iterable de (frame, timestamp_s) avec reconnexion automatique pour les flux live.

    - fichier : timestamp = index / fps (deterministe)
    - flux live (RTSP/USB/IP) : timestamp = horloge monotone
    """

    def __init__(self, source: Union[str, int], reconnect: bool = True,
                 max_retries: int = 5, retry_delay_s: float = 2.0):
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self.source = source
        self.is_file = isinstance(source, str) and os.path.isfile(source)
        self.reconnect = reconnect and not self.is_file
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s
        self.fps = 30.0
        if isinstance(source, str) and source.startswith("rtsp"):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        self._cap = None
        self._open()

    def _open(self) -> None:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise IOError(f"Impossible d'ouvrir la source video: {self.source!r}")
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1 else 30.0

    def __iter__(self) -> Iterator[Tuple[np.ndarray, float]]:
        idx, start, retries = 0, time.monotonic(), 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                if self.is_file or not self.reconnect or retries >= self.max_retries:
                    return
                retries += 1
                log.warning("Flux interrompu, reconnexion %d/%d", retries, self.max_retries)
                time.sleep(self.retry_delay_s)
                self._cap.release()
                try:
                    self._open()
                except IOError:
                    continue
                continue
            retries = 0
            ts = idx / self.fps if self.is_file else time.monotonic() - start
            idx += 1
            yield frame, ts

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
