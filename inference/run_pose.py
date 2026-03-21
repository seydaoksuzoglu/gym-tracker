import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.analysis.squat_analyzer import LiveSquatAnalyzer
from src.vis.skeleton_drawer import draw_landmarks_on_image
from src.sources.webcam import webcam_frames
from src.sources.video import video_frames


def draw_boxed_lines(img, lines, x, y, color, font_scale=0.8, thickness=2, line_gap=30):
    """
    Verilen satırları siyah kutu içine yazar.
    Görüntüyü yerinde günceller ve geri döndürür.
    """
    if not lines:
        return img

    max_width = 0
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
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
            thickness,
            cv2.LINE_AA,
        )
        yy += line_gap

    return img


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
    parser.add_argument(
        "--model",
        default=r"C:\Users\Seyda\Desktop\Projects\GymTracker\models\pose_landmarker_full.task"
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--scale", type=float, default=0.6)
    args = parser.parse_args()

    landmarker = create_landmarker(args.model)
    analyzer = LiveSquatAnalyzer()

    writer = None
    output_path = None
    output_fps = 30.0

    last_live_warnings = []
    live_warning_until_ms = 0

    last_rep_feedback = None
    rep_feedback_until_ms = 0

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        output_path = args.output

    # Kaynak ayarları
    if args.mode == "webcam":
        frame_iter = webcam_frames(args.index)
        display_wait_ms = 1
        output_fps = 30.0
    else:
        if not args.path:
            raise ValueError("--mode video için --path zorunlu.")

        cap_tmp = cv2.VideoCapture(args.path)
        fps = cap_tmp.get(cv2.CAP_PROP_FPS)
        cap_tmp.release()

        output_fps = fps if fps and fps > 1 else 30.0
        display_wait_ms = max(1, int(1000.0 / output_fps))
        frame_iter = video_frames(args.path)

    try:
        for frame_bgr, ts in frame_iter:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, ts)

            analysis = analyzer.analyze(result, ts)
            # Yeni rep başladıysa eski rep sonucunu ekrandan kaldır
            if analysis.state == "descent":
                last_rep_feedback = None
                rep_feedback_until_ms = 0

            if analysis.live_warnings:
                last_live_warnings = analysis.live_warnings
                live_warning_until_ms = ts + 1200

            if analysis.rep_feedback is not None:
                last_rep_feedback = analysis.rep_feedback
                rep_feedback_until_ms = ts + 700

            # Skeleton çiz
            annotated_rgb = draw_landmarks_on_image(frame_rgb, result)
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            # Canlı uyarılar
            if ts < live_warning_until_ms and last_live_warnings:
                lines = [f"UYARI: {w}" for w in last_live_warnings]
                annotated_bgr = draw_boxed_lines(
                    annotated_bgr,
                    lines,
                    x=10,
                    y=40,
                    color=(0, 165, 255),  # turuncu
                    font_scale=0.8,
                    thickness=2,
                    line_gap=32
                )

            # Rep sonucu
            if ts < rep_feedback_until_ms and last_rep_feedback is not None:
                if last_rep_feedback.has_error:
                    lines = [f"REP {last_rep_feedback.rep_count} HATALI"]
                    lines += [f"HATA: {lab}" for lab in last_rep_feedback.error_labels]
                    color = (0, 0, 255)
                else:
                    lines = [f"REP {last_rep_feedback.rep_count} DOGRU FORM"]
                    color = (0, 255, 0)

                annotated_bgr = draw_boxed_lines(
                    annotated_bgr,
                    lines,
                    x=10,
                    y=160,
                    color=color,
                    font_scale=0.85,
                    thickness=2,
                    line_gap=32
                )

            # Anlık rep sayısı ve state
            cv2.putText(
                annotated_bgr,
                f"REP: {analyzer.counter.rep_count}",
                (10, annotated_bgr.shape[0] - 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Küçük debug satırı
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
                2,
                cv2.LINE_AA,
            )

            # Writer
            if writer is None and output_path:
                h, w = annotated_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, output_fps, (w, h))

            if writer:
                writer.write(annotated_bgr)

            # Ekranda gösterim
            disp_w = max(1, int(annotated_bgr.shape[1] * args.scale))
            disp_h = max(1, int(annotated_bgr.shape[0] * args.scale))
            display_frame = cv2.resize(annotated_bgr, (disp_w, disp_h))

            cv2.imshow("MediaPipe Pose Landmarker + Rule Based Squat Analysis", display_frame)

            key = cv2.waitKey(display_wait_ms) & 0xFF
            if key == ord("q"):
                break

    finally:
        if writer:
            writer.release()

        landmarker.close()
        cv2.destroyAllWindows()

        # Video modu bittikten sonra isterse son frame'i görmek için kısa bekleme yapılabilir
        if args.mode == "video":
            cv2.waitKey(1)


if __name__ == "__main__":
    main()