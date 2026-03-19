import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import argparse
import cv2
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.analysis.squat_rules import analyze_frame_live
from src.vis.skeleton_drawer import draw_landmarks_on_image
from src.sources.webcam import webcam_frames
from src.sources.video import video_frames

def draw_boxed_lines(img, lines, x, y, color, font_scale=0.8, thickness=2, line_gap=30):
    if not lines:
        return

    max_width = 0
    sizes = []
    for line in lines:
        (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        sizes.append((w, h))
        max_width = max(max_width, w)

    total_h = len(lines) * line_gap + 10

    cv2.rectangle(
        img,
        (x - 10, y - 30),
        (x + max_width + 20, y - 30 + total_h),
        (0, 0, 0),
        -1
    )

    yy = y
    for line in lines:
        cv2.putText(
            img,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness
        )
        yy += line_gap

def create_landmarker(model_path: str):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.65,
        min_pose_presence_confidence=0.65,
        min_tracking_confidence=0.70,
    )
    return vision.PoseLandmarker.create_from_options(options)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["webcam", "video"], required=True)
    parser.add_argument("--model", default=r"C:\Users\Seyda\Desktop\Projects\GymTracker\models\pose_landmarker_full.task")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--scale", type=float, default=0.6)
    args = parser.parse_args()

    landmarker = create_landmarker(args.model)

    if args.mode == "webcam":
        frame_iter = webcam_frames(args.index)
        display_wait_ms = 1
    else:
        if not args.path:
            raise ValueError("--mode video için --path zorunlu.")
        # Video’nun gerçek FPS’ini oku, waitKey süresini ona göre ayarla
        _cap_tmp = cv2.VideoCapture(args.path)
        _fps = _cap_tmp.get(cv2.CAP_PROP_FPS)
        _cap_tmp.release()
        display_wait_ms = max(1, int(1000.0 / _fps)) if _fps and _fps > 1 else 33
        frame_iter = video_frames(args.path)

    writer = None
    output_path = None

    last_live_warnings = []
    live_warning_until_ms = 0

    last_rep_feedback = None
    rep_feedback_until_ms = 0

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        output_path = args.output

    try:
        for frame_bgr, ts in frame_iter:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, ts)

            analysis = analyze_frame_live(result, ts)

            if analysis is not None:
                if analysis.live_warnings:
                    last_live_warnings = analysis.live_warnings
                    live_warning_until_ms = ts + 1200

                if analysis.rep_feedback is not None:
                    last_rep_feedback = analysis.rep_feedback
                    rep_feedback_until_ms = ts + 2200

            annotated_rgb = draw_landmarks_on_image(frame_rgb, result)
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            # Canli uyarilar
            if ts < live_warning_until_ms and last_live_warnings:
                lines = [f"UYARI: {w}" for w in last_live_warnings]
                draw_boxed_lines(
                    annotated_bgr,
                    lines,
                    x=10,
                    y=40,
                    color=(0, 165, 255),   # turuncu
                    font_scale=0.8,
                    thickness=2,
                    line_gap=32
                )

            # Rep sonucu
            if ts < rep_feedback_until_ms and last_rep_feedback is not None:
                if last_rep_feedback.has_error:
                    lines = [f"REP {last_rep_feedback.rep_count} TAMAMLANDI"]
                    lines += [f"HATA: {lab}" for lab in last_rep_feedback.error_labels]
                    color = (0, 0, 255)
                else:
                    lines = [f"REP {last_rep_feedback.rep_count} DOGRU FORM"]
                    color = (0, 255, 0)

                draw_boxed_lines(
                    annotated_bgr,
                    lines,
                    x=10,
                    y=160,
                    color=color,
                    font_scale=0.85,
                    thickness=2,
                    line_gap=32
                )

            # kucuk debug satiri
            if analysis is not None:
                knee_txt = "None" if analysis.avg_knee_angle is None else f"{analysis.avg_knee_angle:.1f}"
                hip_txt = "None" if analysis.avg_hip_angle is None else f"{analysis.avg_hip_angle:.1f}"
                lean_txt = "None" if analysis.avg_torso_lean_deg is None else f"{analysis.avg_torso_lean_deg:.1f}"

                debug_line = f"STATE={analysis.state} knee={knee_txt} hip={hip_txt} lean={lean_txt}"
                cv2.putText(
                    annotated_bgr,
                    debug_line,
                    (10, annotated_bgr.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            if writer is None and output_path:
                h, w = annotated_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))

            if writer:
                writer.write(annotated_bgr)

            display_frame = cv2.resize(
                annotated_bgr,
                (int(annotated_bgr.shape[1] * args.scale), int(annotated_bgr.shape[0] * args.scale))
            )

            cv2.imshow("MediaPipe Pose Landmarker + Rule Based Squat Analysis", display_frame)
            if cv2.waitKey(display_wait_ms) & 0xFF == ord("q"):
                break

    finally:
        # Video modunda bittikten sonra pencere açık kalsın, tuşa basınca kapansın
        if args.mode == "video":
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        if writer:
            writer.release()
        landmarker.close()


if __name__ == "__main__":
    main()