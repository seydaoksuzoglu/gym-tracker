"""
Katman 2 - Landmark filtreleri (EMA / median).
Ham landmark'i asla feature extractor'a verme; daima filtreden geçir. 
"""
from collections import deque
from typing import List, Optional

from src.analysis.deadlift.landmarks import (
    Landmark,
    LandmarkFrame,
    NUM_LANDMARKS,
)

class EMAFilter:
    """Exponential Moving Average. new = alpha*current + (1-alpha)*prev"""

    def __init__(self, alpha: float = 0.3):
        if not 0 < alpha <= 1:
            raise ValueError("alpha (0, 1] aralığında olmalı.")
        self.alpha = alpha
        self._state: List[Optional[Landmark]] = [None] * NUM_LANDMARKS
    
    def update(self, frame: LandmarkFrame) -> LandmarkFrame:
        out: List[Landmark] = []
        for idx in range(NUM_LANDMARKS):
            current = frame.landmarks[idx]
            if not current.valid:
                prev = self._state[idx]
                out.append(prev if prev is not None else Landmark.missing())
                continue

            prev = self._state[idx]
            if prev is None or not prev.valid:
                filtered = Landmark(
                    x=current.x, y=current.y,
                    visibility=current.visibility, valid=True,
                )
            else:
                filtered = Landmark(
                    x=self.alpha * current.x + (1 - self.alpha) * prev.x,
                    y=self.alpha * current.y + (1 - self.alpha) * prev.y,
                    visibility=current.visibility,
                    valid=True,
                )
            self._state[idx] = filtered
            out.append(filtered)
        return LandmarkFrame(landmarks=out, ts_ms=frame.ts_ms)

class MedianFilter:
    """Slinding-window median filter (her landmark için ayri x, y medyan)."""

    def __init__(self, window: int = 5):
        if window < 1 or window % 2 == 0:
            raise ValueError("window pozitif tek sayi olmali.")
        self.window = window
        self._history_x: List[deque] = [deque(maxlen=window) for _ in range(NUM_LANDMARKS)]
        self._history_y: List[deque] = [deque(maxlen=window) for _ in range(NUM_LANDMARKS)]

    def update(self, frame: LandmarkFrame) -> LandmarkFrame:
        out: List[Landmark] = []
        for idx in range(NUM_LANDMARKS):
            current = frame.landmarks[idx]
            if not current.valid:
                if self._history_x[idx]:
                    last_x = self._history_x[idx][-1]
                    last_y = self._history_y[idx][-1]
                    out.append(Landmark(x=last_x, y=last_y,
                                        visibility=current.visibility, valid=True))
                else:
                    out.append(Landmark.missing())
                continue

            self._history_x[idx].append(current.x)
            self._history_y[idx].append(current.y)
            xs = sorted(self._history_x[idx])
            ys = sorted(self._history_y[idx])
            mid = len(xs) // 2
            out.append(Landmark(
                x=xs[mid], y=ys[mid],
                visibility=current.visibility, valid=True,
            ))
        return LandmarkFrame(landmarks=out, ts_ms=frame.ts_ms)