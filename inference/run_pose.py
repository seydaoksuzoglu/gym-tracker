import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import time
import argparse
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ultralytics import YOLO
from src.pose_backends.yolo26_adapter import extract_yolo_pose_frame
from src.analysis.squat_features_yolo import extract_squat_features_yolo

from src.analysis.squat_analyzer import LiveSquatAnalyzer
from src.vis.skeleton_drawer import (
    draw_landmarks_on_image_mediapipe,
    draw_landmarks_on_image_yolo,
)
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
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--backend", choices=["mediapipe", "yolo26"], default="mediapipe")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--benchmark", action="store_true")

    args = parser.parse_args()
    device = args.device
    landmarker = None
    yolo_model = None
    analyzer = LiveSquatAnalyzer()

    if args.backend == "mediapipe":
        landmarker = create_landmarker(args.model)
        
    else:
        yolo_model = YOLO("yolo26m-pose.pt")

    writer = None
    output_path = None

    frame_count = 0

    total_pipeline_time = 0.0
    total_infer_time = 0.0
    total_analysis_time = 0.0
    total_draw_time = 0.0

    benchmark_start = None

    last_live_warnings = []
    live_warning_until_ms = 0

    last_rep_feedback = None
    rep_feedback_until_ms = 0

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        output_path = args.output

    source_fps = None

    # Kaynak ayarları
    if args.mode == "webcam":
        frame_iter = webcam_frames(args.index)
        display_wait_ms = 1
    else:
        if not args.path:
            raise ValueError("--mode video için --path zorunlu.")
        _cap_tmp = cv2.VideoCapture(args.path)
        _fps = _cap_tmp.get(cv2.CAP_PROP_FPS)
        _cap_tmp.release()
        source_fps = _fps if _fps and _fps > 1 else None
        display_wait_ms = max(1, int(1000.0 / _fps)) if _fps and _fps > 1 else 33
        frame_iter = video_frames(args.path)

    if args.benchmark:
        benchmark_start = time.perf_counter()    
    try:
        for frame_bgr, ts in frame_iter:
            frame_t0 = time.perf_counter()
            frame_count += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            infer_t0 = time.perf_counter()
            if args.backend == "mediapipe":
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result = landmarker.detect_for_video(mp_image, ts)
                infer_t1 = time.perf_counter()

                analysis_t0 = time.perf_counter()
                analysis = analyzer.analyze(result, ts)
                analysis_t1 = time.perf_counter()

            else:
                yolo_results = yolo_model(frame_bgr, device=device, verbose=False)
                infer_t1 = time.perf_counter()

                analysis_t0 = time.perf_counter()
                analysis = analyzer.analyze_yolo(yolo_results[0], ts)
                analysis_t1 = time.perf_counter()

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
            draw_t0 = time.perf_counter()

            if args.backend == "mediapipe":
                annotated_rgb = draw_landmarks_on_image_mediapipe(frame_rgb, result)
            else:
                annotated_rgb = draw_landmarks_on_image_yolo(frame_rgb, yolo_results[0])

            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            draw_t1 = time.perf_counter()

            total_infer_time += (infer_t1 - infer_t0)
            total_analysis_time += (analysis_t1 - analysis_t0)
            total_draw_time += (draw_t1 - draw_t0)

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
                output_fps = 30
                writer = cv2.VideoWriter(output_path, fourcc, output_fps, (w, h))

            if writer:
                writer.write(annotated_bgr)

            frame_t1 = time.perf_counter()
            total_pipeline_time += (frame_t1 - frame_t0)

            if not args.benchmark:
                display_frame = cv2.resize(
                    annotated_bgr,
                    (int(annotated_bgr.shape[1] * args.scale), int(annotated_bgr.shape[0] * args.scale))
                )

                cv2.imshow("Pose Estimation Based Squat Analysis", display_frame)
                if cv2.waitKey(display_wait_ms) & 0xFF == ord("q"):
                    break

    finally:
        if args.benchmark and frame_count > 0:
            total_elapsed = time.perf_counter() - benchmark_start if benchmark_start is not None else total_pipeline_time

            avg_pipeline_ms = (total_pipeline_time / frame_count) * 1000.0
            avg_infer_ms = (total_infer_time / frame_count) * 1000.0
            avg_analysis_ms = (total_analysis_time / frame_count) * 1000.0
            avg_draw_ms = (total_draw_time / frame_count) * 1000.0

            pipeline_fps = frame_count / total_pipeline_time if total_pipeline_time > 0 else 0.0
            wall_fps = frame_count / total_elapsed if total_elapsed > 0 else 0.0

            print("\n===== OFFLINE VIDEO BENCHMARK =====")
            print(f"Frames processed       : {frame_count}")
            print(f"Total elapsed (wall)   : {total_elapsed:.3f} s")
            print(f"Total pipeline time    : {total_pipeline_time:.3f} s")
            print(f"Average pipeline/frame : {avg_pipeline_ms:.2f} ms")
            print(f"Average inference      : {avg_infer_ms:.2f} ms")
            print(f"Average analysis       : {avg_analysis_ms:.2f} ms")
            print(f"Average drawing        : {avg_draw_ms:.2f} ms")
            print(f"Pipeline FPS           : {pipeline_fps:.2f}")
            print(f"Wall-clock FPS         : {wall_fps:.2f}")

            if source_fps is not None:
                realtime_ratio = pipeline_fps / source_fps if source_fps > 0 else 0.0
                print(f"Source video FPS       : {source_fps:.2f}")
                print(f"Real-time ratio        : {realtime_ratio:.2f}x")

        if writer:
            writer.release()

        if landmarker is not None:
            landmarker.close()
        cv2.destroyAllWindows()

        # Video modu bittikten sonra isterse son frame'i görmek için kısa bekleme yapılabilir
        if args.mode == "video" and not args.benchmark:
            cv2.waitKey(0)


if __name__ == "__main__":
    main()