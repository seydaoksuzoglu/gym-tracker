"""StandingCalibrator entegrasyon testi (fixture video gerektirir)."""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import cv2
import pytest

from src.analysis.deadlift.calibration import StandingCalibrator
from src.analysis.deadlift.capture import PoseCapture
from src.analysis.deadlift.filters import EMAFilter
from src.sources.video import video_frames


FIXTURE_VIDEO = REPO_ROOT / "tests" / "fixtures" / "deadlift_001.mp4"
EXPECTED_BASELINE = REPO_ROOT / "tests" / "fixtures" / "expected_deadlift_001.json"
MODEL_PATH = REPO_ROOT / "models" / "pose_landmarker_full.task"

TOLERANCE = 0.05  # +-%5


def test_calibration_matches_expected():
    if not FIXTURE_VIDEO.exists():
        pytest.skip(f"Fixture video bulunamadi: {FIXTURE_VIDEO}")
    if not MODEL_PATH.exists():
        pytest.skip(f"Pose model bulunamadi: {MODEL_PATH}")

    capture = PoseCapture(model_path=str(MODEL_PATH))
    ema = EMAFilter(alpha=0.3)
    calib = StandingCalibrator(frames_required=30)

    try:
        for frame_bgr, ts in video_frames(str(FIXTURE_VIDEO)):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            raw = capture.process(frame_rgb, ts)
            filtered = ema.update(raw)
            calib.update(filtered)
            if calib.is_ready() or calib.has_failed():
                break
    finally:
        capture.close()

    if calib.has_failed():
        pytest.fail("Kalibrasyon basarisiz (hareketli kullanici?). Fixture videoyu kontrol et.")

    assert calib.is_ready(), "Kalibrasyon 30 frame'de tamamlanamadi"
    actual = calib.get_baseline()
    actual_dict = {
        "torso_length": actual.torso_length,
        "shoulder_width": actual.shoulder_width,
        "femur_length": actual.femur_length,
    }

    # Snapshot olusturma modu
    if os.environ.get("UPDATE_SNAPSHOT") == "1":
        EXPECTED_BASELINE.parent.mkdir(parents=True, exist_ok=True)
        with open(EXPECTED_BASELINE, "w", encoding="utf-8") as f:
            json.dump(actual_dict, f, indent=2)
        print(f"\n[snapshot] {EXPECTED_BASELINE} yazildi: {actual_dict}")
        return

    if not EXPECTED_BASELINE.exists():
        pytest.fail(
            f"Beklenen baseline JSON bulunamadi: {EXPECTED_BASELINE}\n"
            f"Ilk kez kosturuyorsan: $env:UPDATE_SNAPSHOT='1'; python -m pytest {Path(__file__).name} -s"
        )

    with open(EXPECTED_BASELINE, "r", encoding="utf-8") as f:
        expected = json.load(f)

    for key in ("torso_length", "shoulder_width", "femur_length"):
        actual_val = actual_dict[key]
        expected_val = expected[key]
        rel_err = abs(actual_val - expected_val) / max(abs(expected_val), 1e-9)
        assert rel_err <= TOLERANCE, (
            f"{key}: actual={actual_val:.4f}, expected={expected_val:.4f}, "
            f"rel_err={rel_err:.4%} > {TOLERANCE:.0%}"
        )
