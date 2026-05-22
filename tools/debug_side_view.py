"""
Tek video uzerinde side-view ratio'sunu ilk N frame icin print eder.
Kullanim: python tools/debug_side_view.py tests/fixtures/validation/deadlift_002.mp4
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2

from src.analysis.deadlift.capture import PoseCapture
from src.analysis.deadlift.filters import EMAFilter
from src.analysis.deadlift.calibration import StandingCalibrator
from src.analysis.deadlift.landmarks import LEFT, RIGHT


def main():
    video_path = Path(sys.argv[1])
    capture = PoseCapture("models/pose_landmarker_full.task", running_mode="VIDEO")
    ema = EMAFilter(alpha=0.3)
    calibrator = StandingCalibrator(frames_required=30)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"{video_path.name}: fps={fps}")

    baseline = None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame_idx > 80:
            break
        ts_ms = int(frame_idx * 1000 / fps)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw = capture.process(frame_rgb, ts_ms)
        filtered = ema.update(raw)

        if not calibrator.is_ready():
            calibrator.update(filtered)
            if calibrator.is_ready():
                baseline = calibrator.get_baseline()
                print(f"\nKalibrasyon tamamlandi @ frame {frame_idx}:")
                print(f"  torso_length    = {baseline.torso_length:.4f}")
                print(f"  shoulder_width  = {baseline.shoulder_width:.4f}")
                print(f"  femur_length    = {baseline.femur_length:.4f}")
                print(f"  threshold (0.20 * torso) = {baseline.torso_length * 0.20:.4f}\n")
                print(f"{'frame':>6} {'left_x':>8} {'right_x':>8} {'|dx|':>8} {'ratio':>8} {'verdict':>10}")
                print("-" * 60)
            frame_idx += 1
            continue

        ls = filtered.get(LEFT["shoulder"])
        rs = filtered.get(RIGHT["shoulder"])
        if ls.valid and rs.valid:
            dx = abs(ls.x - rs.x)
            ratio = dx / baseline.torso_length
            verdict = "REJECT" if ratio > 0.20 else "ok"
            print(f"{frame_idx:>6} {ls.x:>8.4f} {rs.x:>8.4f} {dx:>8.4f} {ratio:>8.3f} {verdict:>10}")
        else:
            print(f"{frame_idx:>6}  missing shoulders")
        frame_idx += 1

    capture.close()
    cap.release()


if __name__ == "__main__":
    main()
