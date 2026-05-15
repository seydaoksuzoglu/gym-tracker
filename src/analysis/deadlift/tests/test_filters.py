"""EMAFilter ve MedianFilter unit testleri."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import pytest
from src.analysis.deadlift.filters import EMAFilter, MedianFilter
from src.analysis.deadlift.landmarks import (
    Landmark, LandmarkFrame, NUM_LANDMARKS,
)


def _frame(positions: dict, ts_ms: int = 0) -> LandmarkFrame:
    landmarks = [Landmark.missing() for _ in range(NUM_LANDMARKS)]
    for idx, (x, y) in positions.items():
        landmarks[idx] = Landmark(x=x, y=y, visibility=1.0, valid=True)
    return LandmarkFrame(landmarks=landmarks, ts_ms=ts_ms)


def test_ema_first_frame_passes_through():
    f = EMAFilter(alpha=0.5)
    out = f.update(_frame({0: (1.0, 2.0)}))
    assert out.landmarks[0].x == pytest.approx(1.0)
    assert out.landmarks[0].y == pytest.approx(2.0)


def test_ema_smooths_step_input():
    f = EMAFilter(alpha=0.5)
    out1 = f.update(_frame({0: (0.0, 0.0)}))
    out2 = f.update(_frame({0: (1.0, 1.0)}))
    out3 = f.update(_frame({0: (1.0, 1.0)}))
    assert out1.landmarks[0].x == pytest.approx(0.0)
    assert out2.landmarks[0].x == pytest.approx(0.5)
    assert out3.landmarks[0].x == pytest.approx(0.75)


def test_ema_skips_invalid():
    f = EMAFilter(alpha=0.5)
    f.update(_frame({0: (0.0, 0.0)}))
    out = f.update(_frame({}))
    assert out.landmarks[0].x == pytest.approx(0.0)
    assert out.landmarks[0].valid


def test_median_basic():
    f = MedianFilter(window=3)
    f.update(_frame({0: (1.0, 1.0)}))
    f.update(_frame({0: (5.0, 5.0)}))
    out = f.update(_frame({0: (3.0, 3.0)}))
    assert out.landmarks[0].x == pytest.approx(3.0)


def test_median_outlier_robustness():
    f = MedianFilter(window=5)
    out = None
    for x in [1.0, 1.0, 100.0, 1.0, 1.0]:
        out = f.update(_frame({0: (x, x)}))
    assert out.landmarks[0].x == pytest.approx(1.0)


def test_invalid_alpha_rejected():
    with pytest.raises(ValueError):
        EMAFilter(alpha=0.0)
    with pytest.raises(ValueError):
        EMAFilter(alpha=1.5)


def test_invalid_window_rejected():
    with pytest.raises(ValueError):
        MedianFilter(window=4)
    with pytest.raises(ValueError):
        MedianFilter(window=0)
