"""pre_checks.check_side_view birim testleri."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.deadlift.calibration import StandingBaseline
from src.analysis.deadlift.landmarks import (
    LEFT, RIGHT, Landmark, LandmarkFrame, NUM_LANDMARKS,
)
from src.analysis.deadlift.pre_checks import check_side_view


def _frame_with_shoulders(left_x: float, right_x: float) -> LandmarkFrame:
    landmarks = [Landmark.missing() for _ in range(NUM_LANDMARKS)]
    landmarks[LEFT["shoulder"]] = Landmark(left_x, 0.4, 1.0, True)
    landmarks[RIGHT["shoulder"]] = Landmark(right_x, 0.4, 1.0, True)
    return LandmarkFrame(landmarks=landmarks, ts_ms=0)


BASELINE = StandingBaseline(torso_length=0.3, shoulder_width=0.20, femur_length=0.25)
# threshold = 0.20 * 0.30 = 0.06


def test_passes_when_baseline_none():
    frame = _frame_with_shoulders(0.40, 0.45)
    assert check_side_view(frame, baseline=None) is None


def test_side_view_accepted():
    # 0.04 < 0.06 -> side
    frame = _frame_with_shoulders(0.48, 0.52)
    assert check_side_view(frame, baseline=BASELINE) is None


def test_front_view_rejected():
    # 0.18 > 0.06 -> front
    frame = _frame_with_shoulders(0.30, 0.48)
    rejected = check_side_view(frame, baseline=BASELINE)
    assert rejected is not None
    assert rejected.reason == "not_side_view"


def test_missing_shoulders_rejected():
    landmarks = [Landmark.missing() for _ in range(NUM_LANDMARKS)]
    frame = LandmarkFrame(landmarks=landmarks, ts_ms=0)
    rejected = check_side_view(frame, baseline=BASELINE)
    assert rejected is not None
    assert rejected.reason == "missing_shoulders"
