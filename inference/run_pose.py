#  python inference/run_pose.py --exercise deadlift --mode video --path tests/fixtures/validation/deadlift_001.mp4 --scale 0.6
# python inference/run_pose.py --exercise deadlift --mode video --path tests/fixtures/deadlift_012.mp4 --scale 0.6
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

from src.analysis.squat.squat_analyzer import LiveSquatAnalyzer
from src.vis.skeleton_drawer import (
    draw_landmarks_on_image_mediapipe,
    draw_landmarks_on_image_yolo,
    draw_phase_label,
    draw_rep_counter,
)
from src.sources.webcam import webcam_frames
from src.sources.video import video_frames


# Repo kökünü tek noktada bul
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "pose_landmarker_full.task"

def make_analyzer(exercise: str, model_path: str):
    """Egzersiz tipine gore uygun analyzer'i dondurur."""
    if exercise == "squat":
        return LiveSquatAnalyzer()
    if exercise == "deadlift":
        # Lazy import: squat kullanicisi deadlift modulunu yuklemez
        from src.analysis.deadlift.deadlift_analyzer import LiveDeadliftAnalyzer
        return LiveDeadliftAnalyzer(model_path=model_path)
    raise ValueError(f"Bilinmeyen egzersiz: {exercise}")

def _run_deadlift(args, analyzer, frame_iter, display_wait_ms, output_path):
    writer = None
    _frame_idx = 0
    # Sprint 4 demo: cumulative error sayaclari + rep feedback (1.5 sn gozukur)
    error_counts = {"incomplete_lockout": 0, "uncontrolled_descent": 0}
    last_rep_feedback = None
    rep_feedback_until_ms = 0
    GRADE_COLORS = {
        "perfect": (0, 255, 0),
        "good": (100, 220, 100),
        "needs_attention": (0, 200, 255),
        "form_issue": (0, 100, 255),
        "severe": (0, 0, 255),
    }
    try:
        for frame_bgr, ts in frame_iter:
            _frame_idx += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            analysis = analyzer.analyze(frame_rgb, ts)

            annotated = frame_bgr.copy()

            # Skeleton (varsa)
            if analysis.landmarks is not None:
                h, w = annotated.shape[:2]
                conns = [(11,23),(12,24),(23,25),(24,26),(25,27),(26,28),(11,12),(23,24)]
                for lm in analysis.landmarks.landmarks:
                    if lm.valid:
                        cv2.circle(annotated, (int(lm.x*w), int(lm.y*h)), 4, (0,255,0), -1)
                for a,b in conns:
                    la, lb = analysis.landmarks.landmarks[a], analysis.landmarks.landmarks[b]
                    if la.valid and lb.valid:
                        cv2.line(annotated,
                                 (int(la.x*w), int(la.y*h)),
                                 (int(lb.x*w), int(lb.y*h)),
                                 (0,255,0), 2)

            # Durum overlay
            if analysis.calibrating:
                cv2.putText(annotated, "CALIBRATING... stand still",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
            elif analysis.rejected_reason:
                cv2.putText(annotated, f"REJECTED: {analysis.rejected_reason}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            else:
                draw_phase_label(annotated, analysis.phase, ts)
                draw_rep_counter(annotated, analysis.rep_count, analysis.incomplete_count)
                t_txt = "None" if analysis.T is None else f"{analysis.T:.1f}"
                dy_txt = "None" if analysis.hip_dy is None else f"{analysis.hip_dy:+.3f}"
                v_txt = "None" if analysis.hip_velocity is None else f"{analysis.hip_velocity:+.5f}"
                cand_txt = (
                    "None" if analysis.candidate_phase is None
                    else f"{analysis.candidate_phase}({analysis.candidate_held_ms}ms)"
                )
                if _frame_idx % 3 == 0:
                    print(f"[{ts:6d}ms] phase={analysis.phase:8s} T={t_txt:>5s} "
                        f"dy={dy_txt:>7s} v={v_txt:>10s} cand={cand_txt}")
                cv2.putText(annotated, f"T={t_txt} dy={dy_txt} v={v_txt} cand={cand_txt}",
                            (10, annotated.shape[0]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

                # Sprint 4: cumulative error sayaci (sag ust)
                err_txt = (
                    f"ILO:{error_counts['incomplete_lockout']}  "
                    f"UCD:{error_counts['uncontrolled_descent']}"
                )
                (tw, _), _ = cv2.getTextSize(err_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.putText(annotated, err_txt,
                            (annotated.shape[1] - tw - 10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 2)

            if analysis.rep_event:
                if analysis.rep_event.rep_completed:
                    print(f"[{ts:6d}ms] *** REP {analysis.rep_event.rep_id} TAMAMLANDI *** "
                        f"grade={analysis.rep_overall_grade} "
                        f"durations={analysis.rep_event.phase_durations_ms}")
                elif analysis.rep_event.incomplete:
                    print(f"[{ts:6d}ms] !!! INCOMPLETE REP — reason={analysis.rep_event.incomplete_reason} "
                        f"durations={analysis.rep_event.phase_durations_ms}")

            # Sprint 4: rep kapaninca grade + hata sayaclari
            if analysis.rep_event and analysis.rep_event.rep_completed:
                errors = analysis.rep_errors or {}
                flagged = []
                for eid, report in errors.items():
                    if report.confidence >= 0.3:   # UI esigi
                        error_counts[eid] = error_counts.get(eid, 0) + 1
                        flagged.append(eid)
                grade = analysis.rep_overall_grade or "?"
                last_rep_feedback = {
                    "rep_id": analysis.rep_event.rep_id,
                    "grade": grade,
                    "flagged": flagged,
                    "color": GRADE_COLORS.get(grade, (255, 255, 255)),
                }
                rep_feedback_until_ms = ts + 1500

            # Rep feedback overlay (1.5 sn boyunca)
            if last_rep_feedback and ts < rep_feedback_until_ms:
                lines = [f"REP {last_rep_feedback['rep_id']}: {last_rep_feedback['grade'].upper()}"]
                for eid in last_rep_feedback["flagged"]:
                    lines.append(f"  - {eid}")
                y0 = 160
                for i, line in enumerate(lines):
                    cv2.putText(annotated, line, (10, y0 + i * 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                last_rep_feedback["color"], 2)


            # Output writer
            if writer is None and output_path:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
            if writer:
                writer.write(annotated)

            if not args.benchmark:
                disp = cv2.resize(
                    annotated,
                    (int(annotated.shape[1]*args.scale), int(annotated.shape[0]*args.scale)))
                cv2.imshow("Pose Estimation Based Analysis", disp)
                if cv2.waitKey(display_wait_ms) & 0xFF == ord("q"):
                    break
    finally:
        if writer:
            writer.release()
        analyzer.close()
        cv2.destroyAllWindows()
        if args.mode == "video" and not args.benchmark:
            cv2.waitKey(0)


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
    parser.add_argument("--exercise", choices=["squat", "deadlift"], default="squat")
    parser.add_argument("--mode", choices=["webcam", "video"], required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--backend", choices=["mediapipe", "yolo26"], default="mediapipe")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--benchmark", action="store_true")

    args = parser.parse_args()
    device = args.device

    # Analyzer factory
    analyzer = make_analyzer(args.exercise, args.model)

    landmarker = None
    yolo_model = None
    # YOLO ve squat YOLO feature'ı SADECE yolo26 backend seçildiğinde import et
    extract_yolo_pose_frame = None
    if args.backend == "mediapipe":
        landmarker = create_landmarker(args.model)
    else:
        from ultralytics import YOLO
        from src.pose_backends.yolo26_adapter import (
            extract_yolo_pose_frame as _extract_yolo,
        )
        extract_yolo_pose_frame = _extract_yolo
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

    # === DEADLIFT KOLU (erken cikis - asagisi calismaz) ===
    if args.exercise == "deadlift":
        if args.backend != "mediapipe":
            raise NotImplementedError("Deadlift V0 sadece mediapipe backend'inde calisir.")
        _run_deadlift(args, analyzer, frame_iter, display_wait_ms, output_path)
        return
    # === SQUAT KOLU (asagisi mevcut kod, dokunulmadi) ===

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

            if analysis.state == "descent":
                last_rep_feedback = None
                rep_feedback_until_ms = 0

            if analysis.live_warnings:
                last_live_warnings = analysis.live_warnings
                live_warning_until_ms = ts + 1200

            if analysis.rep_feedback is not None:
                last_rep_feedback = analysis.rep_feedback
                rep_feedback_until_ms = ts + 700

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

            if ts < live_warning_until_ms and last_live_warnings:
                lines = [f"UYARI: {w}" for w in last_live_warnings]
                annotated_bgr = draw_boxed_lines(
                    annotated_bgr, lines, x=10, y=40,
                    color=(0, 165, 255), font_scale=0.8, thickness=2, line_gap=32,
                )

            if ts < rep_feedback_until_ms and last_rep_feedback is not None:
                if last_rep_feedback.has_error:
                    lines = [f"REP {last_rep_feedback.rep_count} HATALI"]
                    lines += [f"HATA: {lab}" for lab in last_rep_feedback.error_labels]
                    color = (0, 0, 255)
                else:
                    lines = [f"REP {last_rep_feedback.rep_count} DOGRU FORM"]
                    color = (0, 255, 0)

                annotated_bgr = draw_boxed_lines(
                    annotated_bgr, lines, x=10, y=160,
                    color=color, font_scale=0.85, thickness=2, line_gap=32,
                )

            cv2.putText(
                annotated_bgr, f"REP: {analyzer.counter.rep_count}",
                (10, annotated_bgr.shape[0] - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
            )

            knee_txt = "None" if analysis.avg_knee_angle is None else f"{analysis.avg_knee_angle:.1f}"
            hip_txt = "None" if analysis.avg_hip_angle is None else f"{analysis.avg_hip_angle:.1f}"
            lean_txt = "None" if analysis.avg_torso_lean_deg is None else f"{analysis.avg_torso_lean_deg:.1f}"

            debug_line = f"STATE={analysis.state} knee={knee_txt} hip={hip_txt} lean={lean_txt}"
            cv2.putText(
                annotated_bgr, debug_line,
                (10, annotated_bgr.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )

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
                    (int(annotated_bgr.shape[1] * args.scale),
                     int(annotated_bgr.shape[0] * args.scale)),
                )
                cv2.imshow("Pose Estimation Based Analysis", display_frame)
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

        if args.mode == "video" and not args.benchmark:
            cv2.waitKey(0)


if __name__ == "__main__":
    main()