import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from src.config import settings
from src.sources.video import video_frames


# ------------------------------ Result dataclasses ------------------------------

@dataclass
class RepResult:
    rep_index: int
    overall_grade: Optional[str] = None
    phase_durations_ms: Optional[dict] = None
    video_ts_ms: Optional[int] = None
    errors: dict = field(default_factory=dict)

    def to_save_rep_dict(self) -> dict:
        return {
            "rep_id": self.rep_index,
            "overall_grade": self.overall_grade,
            "phase_durations_ms": self.phase_durations_ms,
            "video_ts_ms": self.video_ts_ms,
            "errors": self.errors,
        }


@dataclass
class AnalysisResult:
    exercise: str
    backend: str
    total_frames: int
    total_reps: int
    output_video_path: Optional[str] = None
    rejected_frames: int = 0
    reps: list[RepResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ------------------------------ Public API ------------------------------

def analyze_video(
    path: str,
    exercise: str,
    backend: str = "mediapipe",
    output_video_path: Optional[str] = None,
) -> AnalysisResult:
    if exercise == "deadlift":
        return _analyze_deadlift(path, output_video_path)
    if exercise == "squat":
        return _analyze_squat(path, backend, output_video_path)
    raise ValueError(f"Bilinmeyen egzersiz: {exercise}")


# ------------------------------ Shared helpers ------------------------------

def _open_writer(output_path: Optional[str], shape) -> Optional[cv2.VideoWriter]:
    if output_path is None:
        return None
    h, w = shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, 30, (w, h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(output_path, fourcc, 30, (w, h))
    if not writer.isOpened():
        return None
    return writer

def _transcode_to_h264(src_path: str) -> Optional[str]:
    """mp4v -> H.264 (libx264). Tarayicilar oynatabilsin diye."""
    try:
        import imageio_ffmpeg
        import subprocess
    except ImportError:
        return None

    src = Path(src_path)
    if not src.exists():
        return None

    dst = src.with_name(src.stem + "_h264.mp4")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    try:
        src.unlink()
    except OSError:
        pass
    return str(dst)


def _draw_boxed_lines(img, lines, x, y, color, font_scale=0.7, thickness=2, line_gap=28):
    if not lines:
        return img
    max_width = 0
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        max_width = max(max_width, w)
    total_h = len(lines) * line_gap + 10
    cv2.rectangle(img, (x - 10, y - 28), (x + max_width + 20, y - 28 + total_h), (0, 0, 0), -1)
    yy = y
    for line in lines:
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        yy += line_gap
    return img


# ------------------------------ Deadlift ------------------------------

_DEADLIFT_CONNS = [(11, 23), (12, 24), (23, 25), (24, 26), (25, 27), (26, 28), (11, 12), (23, 24)]
_GRADE_COLORS = {
    "perfect": (0, 255, 0),
    "good": (100, 220, 100),
    "needs_attention": (0, 200, 255),
    "form_issue": (0, 100, 255),
    "severe": (0, 0, 255),
}


def _draw_deadlift_frame(annotated, analysis, ts, error_counts, last_rep_feedback, rep_feedback_until_ms):
    from src.vis.skeleton_drawer import draw_phase_label, draw_rep_counter

    if analysis.landmarks is not None:
        h, w = annotated.shape[:2]
        for lm in analysis.landmarks.landmarks:
            if lm.valid:
                cv2.circle(annotated, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)
        for a, b in _DEADLIFT_CONNS:
            la, lb = analysis.landmarks.landmarks[a], analysis.landmarks.landmarks[b]
            if la.valid and lb.valid:
                cv2.line(annotated,
                         (int(la.x * w), int(la.y * h)),
                         (int(lb.x * w), int(lb.y * h)),
                         (0, 255, 0), 2)

    if analysis.calibrating:
        cv2.putText(annotated, "CALIBRATING... stand still",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    elif analysis.rejected_reason:
        cv2.putText(annotated, f"REJECTED: {analysis.rejected_reason}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        draw_phase_label(annotated, analysis.phase, ts)
        draw_rep_counter(annotated, analysis.rep_count, analysis.incomplete_count)
        err_txt = f"ILO:{error_counts['incomplete_lockout']}  UCD:{error_counts['uncontrolled_descent']}"
        (tw, _), _ = cv2.getTextSize(err_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(annotated, err_txt,
                    (annotated.shape[1] - tw - 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 2)

    if last_rep_feedback and ts < rep_feedback_until_ms:
        lines = [f"REP {last_rep_feedback['rep_id']}: {last_rep_feedback['grade'].upper()}"]
        for eid in last_rep_feedback["flagged"]:
            lines.append(f"  - {eid}")
        y0 = 160
        for i, line in enumerate(lines):
            cv2.putText(annotated, line, (10, y0 + i * 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        last_rep_feedback["color"], 2)


def _analyze_deadlift(path: str, output_video_path: Optional[str]) -> AnalysisResult:
    from src.analysis.deadlift.deadlift_analyzer import LiveDeadliftAnalyzer
    analyzer = LiveDeadliftAnalyzer(model_path=str(settings.mediapipe_model_abs))

    reps: list[RepResult] = []
    total_frames = 0
    rejected_frames = 0
    error_counts = {"incomplete_lockout": 0, "uncontrolled_descent": 0}
    last_rep_feedback = None
    rep_feedback_until_ms = 0
    writer: Optional[cv2.VideoWriter] = None
    notes: list[str] = []

    try:
        for frame_bgr, ts in video_frames(path):
            total_frames += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            analysis = analyzer.analyze(frame_rgb, ts)

            if analysis.rejected_reason:
                rejected_frames += 1

            if analysis.rep_event and analysis.rep_event.rep_completed:
                errors_out: dict[str, dict] = {}
                flagged = []
                for eid, report in (analysis.rep_errors or {}).items():
                    errors_out[eid] = {
                        "area": report.area,
                        "confidence": report.confidence,
                        "evidence": getattr(report, "evidence", None),
                    }
                    if report.confidence >= 0.3:
                        error_counts[eid] = error_counts.get(eid, 0) + 1
                        flagged.append(eid)
                grade = analysis.rep_overall_grade or "?"
                last_rep_feedback = {
                    "rep_id": analysis.rep_event.rep_id,
                    "grade": grade,
                    "flagged": flagged,
                    "color": _GRADE_COLORS.get(grade, (255, 255, 255)),
                }
                rep_feedback_until_ms = ts + 1500
                reps.append(RepResult(
                    rep_index=analysis.rep_event.rep_id,
                    overall_grade=analysis.rep_overall_grade,
                    phase_durations_ms=analysis.rep_event.phase_durations_ms,
                    video_ts_ms=ts,
                    errors=errors_out,
                ))

            if output_video_path is not None:
                annotated = frame_bgr.copy()
                _draw_deadlift_frame(annotated, analysis, ts, error_counts,
                                     last_rep_feedback, rep_feedback_until_ms)
                if writer is None:
                    writer = _open_writer(output_video_path, annotated.shape)
                    if writer is None:
                        notes.append("Video writer acilamadi (codec). Sadece numeric sonuc.")
                        output_video_path = None
                if writer is not None:
                    writer.write(annotated)
    finally:
        if writer is not None:
            writer.release()
        if hasattr(analyzer, "close"):
            analyzer.close()

    if output_video_path is not None:
        transcoded = _transcode_to_h264(output_video_path)
        if transcoded is not None:
            output_video_path = transcoded
        else:
            notes.append("Tarayici uyumlu transcode basarisiz; ham mp4v kullaniliyor.")

    return AnalysisResult(
        exercise="deadlift",
        backend="mediapipe",
        total_frames=total_frames,
        total_reps=len(reps),
        output_video_path=output_video_path,
        rejected_frames=rejected_frames,
        reps=reps,
        notes=notes,
    )


# ------------------------------ Squat ------------------------------

def _analyze_squat(path: str, backend: str, output_video_path: Optional[str]) -> AnalysisResult:
    if backend != "mediapipe":
        raise NotImplementedError("Sprint B'de squat icin sadece mediapipe destekli")

    from src.analysis.squat.squat_analyzer import LiveSquatAnalyzer
    from src.vis.skeleton_drawer import draw_landmarks_on_image_mediapipe

    analyzer = LiveSquatAnalyzer()
    base_options = mp_python.BaseOptions(model_asset_path=str(settings.mediapipe_model_abs))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.65,
        min_pose_presence_confidence=0.65,
        min_tracking_confidence=0.70,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    reps: list[RepResult] = []
    total_frames = 0
    last_rep_count = 0
    last_live_warnings: list = []
    live_warning_until_ms = 0
    last_rep_feedback = None
    rep_feedback_until_ms = 0
    writer: Optional[cv2.VideoWriter] = None
    notes: list[str] = []

    try:
        for frame_bgr, ts in video_frames(path):
            total_frames += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            pose_result = landmarker.detect_for_video(mp_image, ts)
            analysis = analyzer.analyze(pose_result, ts)

            if getattr(analysis, "state", None) == "descent":
                last_rep_feedback = None
                rep_feedback_until_ms = 0

            if getattr(analysis, "live_warnings", None):
                last_live_warnings = analysis.live_warnings
                live_warning_until_ms = ts + 1200

            if getattr(analysis, "rep_feedback", None) is not None:
                last_rep_feedback = analysis.rep_feedback
                rep_feedback_until_ms = ts + 700

            current_count = analyzer.counter.rep_count
            if current_count > last_rep_count:
                fb = getattr(analysis, "rep_feedback", None)
                grade = None
                errors_out: dict[str, dict] = {}
                if fb is not None:
                    grade = "form_issue" if fb.has_error else "good"
                    for lab in getattr(fb, "error_labels", []) or []:
                        errors_out[lab] = {
                            "area": 3 if fb.has_error else 1,
                            "confidence": 1.0,
                        }
                reps.append(RepResult(
                    rep_index=current_count,
                    overall_grade=grade,
                    video_ts_ms=ts,
                    errors=errors_out,
                ))
                last_rep_count = current_count

            if output_video_path is not None:
                annotated_rgb = draw_landmarks_on_image_mediapipe(frame_rgb, pose_result)
                annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

                if ts < live_warning_until_ms and last_live_warnings:
                    annotated_bgr = _draw_boxed_lines(
                        annotated_bgr, [f"UYARI: {w}" for w in last_live_warnings],
                        x=10, y=40, color=(0, 165, 255),
                    )

                if ts < rep_feedback_until_ms and last_rep_feedback is not None:
                    if last_rep_feedback.has_error:
                        lines = [f"REP {last_rep_feedback.rep_count} HATALI"]
                        lines += [f"HATA: {lab}" for lab in last_rep_feedback.error_labels]
                        color = (0, 0, 255)
                    else:
                        lines = [f"REP {last_rep_feedback.rep_count} DOGRU FORM"]
                        color = (0, 255, 0)
                    annotated_bgr = _draw_boxed_lines(
                        annotated_bgr, lines, x=10, y=160, color=color,
                    )

                cv2.putText(annotated_bgr, f"REP: {analyzer.counter.rep_count}",
                            (10, annotated_bgr.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                if writer is None:
                    writer = _open_writer(output_video_path, annotated_bgr.shape)
                    if writer is None:
                        notes.append("Video writer acilamadi (codec). Sadece numeric sonuc.")
                        output_video_path = None
                if writer is not None:
                    writer.write(annotated_bgr)
    finally:
        if writer is not None:
            writer.release()
        landmarker.close()
        
    if output_video_path is not None:
        transcoded = _transcode_to_h264(output_video_path)
        if transcoded is not None:
            output_video_path = transcoded
        else:
            notes.append("Tarayici uyumlu transcode basarisiz; ham mp4v kullaniliyor.")

    return AnalysisResult(
        exercise="squat",
        backend=backend,
        total_frames=total_frames,
        total_reps=last_rep_count,
        output_video_path=output_video_path,
        reps=reps,
        notes=notes,
    )
