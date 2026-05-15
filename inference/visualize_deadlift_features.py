"""
Sprint 1.1 dogrulama aracı.

Pipeline: capture -> filter -> calibration -> pre-check -> features
Her frame'de skeleton + feature değerleri + kalibrasyon durumu overlay'i.

Production analyzer DEĞİL. Sadece görsel sanity check.

Kullanim:
  python inference/visualize_deadlift_features.py --mode video --path tests/fixtures/deadlift_001.mp4
  python inference/visualize_deadlift_features.py --mode webcam
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
from pathlib import Path

import cv2

from src.analysis.deadlift.capture import PoseCapture
from src.analysis.deadlift.filters import EMAFilter
from src.analysis.deadlift.calibration import StandingCalibrator
from src.analysis.deadlift.pre_checks import check_side_view
from src.analysis.deadlift.deadlift_features import extract_deadlift_features
from src.analysis.deadlift.landmarks import LEFT, RIGHT, LandmarkFrame
from src.sources.video import video_frames
from src.sources.webcam import webcam_frames


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "models" / "pose_landmarker_full.task"


SKELETON_CONNECTIONS = [
    # Gövde
    (LEFT["shoulder"], RIGHT["shoulder"]),
    (LEFT["shoulder"], LEFT["hip"]),
    (RIGHT["shoulder"], RIGHT["hip"]),
    (LEFT["hip"], RIGHT["hip"]),
    # Sol kol
    (LEFT["shoulder"], LEFT["elbow"]),
    (LEFT["elbow"], LEFT["wrist"]),
    # Sag kol
    (RIGHT["shoulder"], RIGHT["elbow"]),
    (RIGHT["elbow"], RIGHT["wrist"]),
    # Sol bacak
    (LEFT["hip"], LEFT["knee"]),
    (LEFT["knee"], LEFT["ankle"]),
    (LEFT["ankle"], LEFT["heel"]),
    (LEFT["ankle"], LEFT["foot_index"]),
    # Sag bacak
    (RIGHT["hip"], RIGHT["knee"]),
    (RIGHT["knee"], RIGHT["ankle"]),
    (RIGHT["ankle"], RIGHT["heel"]),
    (RIGHT["ankle"], RIGHT["foot_index"]),
    # Bas
    (LEFT["ear"], LEFT["shoulder"]),
    (RIGHT["ear"], RIGHT["shoulder"]),
]


def draw_skeleton(frame_bgr, lm_frame: LandmarkFrame):
    h, w = frame_bgr.shape[:2]
    for a, b in SKELETON_CONNECTIONS:
        la = lm_frame.get(a)
        lb = lm_frame.get(b)
        if la.valid and lb.valid:
            pa = (int(la.x * w), int(la.y * h))
            pb = (int(lb.x * w), int(lb.y * h))
            cv2.line(frame_bgr, pa, pb, (0, 200, 255), 2)
    for lm in lm_frame.landmarks:
        if lm.valid:
            px = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame_bgr, px, 3, (0, 255, 0), -1)
    return frame_bgr


def draw_text_panel(frame_bgr, lines):
    x, y = 10, 30
    for text, color in lines:
        cv2.rectangle(frame_bgr, (x - 4, y - 22), (x + 420, y + 6), (0, 0, 0), -1)
        cv2.putText(
            frame_bgr, text, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )
        y += 28
    return frame_bgr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["video", "webcam"], default="video")
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    capture = PoseCapture(model_path=args.model)
    ema = EMAFilter(alpha=0.3)
    calib = StandingCalibrator(frames_required=30)

    if args.mode == "video":
        if not args.path:
            raise ValueError("--mode video icin --path zorunlu")
        frames = video_frames(args.path)
        cap_tmp = cv2.VideoCapture(args.path)
        fps = cap_tmp.get(cv2.CAP_PROP_FPS) or 30
        cap_tmp.release()
        wait_ms = max(1, int(1000.0 / fps))
    else:
        frames = webcam_frames(args.index)
        wait_ms = 1

    writer = None

    try:
        for frame_bgr, ts in frames:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            raw = capture.process(frame_rgb, ts)
            filtered = ema.update(raw)

            if not calib.is_ready() and not calib.has_failed():
                calib.update(filtered)

            baseline = calib.get_baseline() if calib.is_ready() else None
            rejected = check_side_view(filtered, baseline) if baseline else None
            features = extract_deadlift_features(filtered)

            annotated = draw_skeleton(frame_bgr.copy(), filtered)

            lines = []
            if calib.has_failed():
                lines.append(("CALIB FAILED - hareketli kullanici, videoyu bastan al", (0, 0, 255)))
            elif calib.is_ready():
                b = baseline
                lines.append((
                    f"CALIB OK  torso={b.torso_length:.3f}  sh_w={b.shoulder_width:.3f}  femur={b.femur_length:.3f}",
                    (0, 255, 0),
                ))
            else:
                progress = len(calib._torso_samples)
                lines.append((f"CALIBRATING  {progress}/30", (0, 255, 255)))

            if rejected is not None:
                lines.append((f"REJECTED: {rejected.reason}", (0, 100, 255)))
            elif baseline is not None:
                lines.append(("VIEW: SIDE OK", (0, 255, 0)))

            if features.valid:
                lines.append((
                    f"T={features.T:5.1f}  K={features.K:5.1f}  back={features.back_angle:5.1f}",
                    (255, 255, 255),
                ))
                lines.append((
                    f"hip_y={features.hip_y:.3f}  side={features.side}",
                    (200, 200, 200),
                ))
            else:
                lines.append(("FEATURES INVALID (low visibility)", (0, 100, 255)))

            annotated = draw_text_panel(annotated, lines)

            if args.output and writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.output, fourcc, 30, (w, h))
            if writer:
                writer.write(annotated)

            if args.scale != 1.0:
                h, w = annotated.shape[:2]
                annotated = cv2.resize(
                    annotated, (int(w * args.scale), int(h * args.scale)),
                )

            cv2.imshow("Deadlift Sprint 1.1 - debug", annotated)
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break
    finally:
        capture.close()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
