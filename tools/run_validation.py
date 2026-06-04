"""
Validation videolarini analyzer'dan gecirir, JSON + overlay'li mp4 ciktilari toplar.
Kullanim: python tools/run_validation.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import json
import cv2

from app._engine import _draw_deadlift_frame, _open_writer, _transcode_to_h264
from src.analysis.deadlift.deadlift_analyzer import LiveDeadliftAnalyzer


FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "validation"
OUTPUT_DIR = FIXTURE_DIR / "system_output"
VIDEO_OUT_DIR = REPO_ROOT / "outputs" / "validation"
MODEL_PATH = str(REPO_ROOT / "models" / "pose_landmarker_full.task")

_GRADE_COLORS = {
    "perfect": (0, 255, 0),
    "good": (100, 220, 100),
    "needs_attention": (0, 200, 255),
    "form_issue": (0, 100, 255),
    "severe": (0, 0, 255),
}


def run_video(video_path: Path, video_out_path: Path | None = None) -> dict:
    analyzer = LiveDeadliftAnalyzer(MODEL_PATH)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {total} frame, {fps:.1f} fps")

    reps = []
    incompletes = []
    error_counts = {"incomplete_lockout": 0, "uncontrolled_descent": 0}
    last_rep_feedback = None
    rep_feedback_until_ms = 0
    writer = None
    raw_video_out = str(video_out_path) if video_out_path else None
    # --- tani sayaclari ---
    diag = {
        "calibrating_frames": 0,
        "calibration_completed_at_frame": None,
        "rejected_frames": 0,
        "rejection_reasons": {},   # reason -> count
        "valid_feature_frames": 0,
        "phase_changes": 0,
        "phases_seen": set(),
    }
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts_ms = int(frame_idx * 1000 / fps)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = analyzer.analyze(frame_rgb, ts_ms)

            # Tani: her frame'in akibetini izle
            if out.calibrating:
                diag["calibrating_frames"] += 1
            elif diag["calibration_completed_at_frame"] is None:
                diag["calibration_completed_at_frame"] = frame_idx

            if out.rejected_reason:
                diag["rejected_frames"] += 1
                diag["rejection_reasons"][out.rejected_reason] = (
                    diag["rejection_reasons"].get(out.rejected_reason, 0) + 1
                )

            if out.T is not None:
                diag["valid_feature_frames"] += 1

            if out.phase:
                diag["phases_seen"].add(out.phase)

            if out.rep_event:
                diag["phase_changes"] += 1

            if out.rep_event and out.rep_event.rep_completed:
                flagged = []
                for eid, r in (out.rep_errors or {}).items():
                    if r.confidence >= 0.3:
                        error_counts[eid] = error_counts.get(eid, 0) + 1
                        flagged.append(eid)
                grade = out.rep_overall_grade or "?"
                last_rep_feedback = {
                    "rep_id": out.rep_event.rep_id,
                    "grade": grade,
                    "flagged": flagged,
                    "color": _GRADE_COLORS.get(grade, (255, 255, 255)),
                }
                rep_feedback_until_ms = ts_ms + 1500
                rep_data = {
                    "rep_id": out.rep_event.rep_id,
                    "phase_durations_ms": dict(out.rep_event.phase_durations_ms or {}),
                    "overall_grade": out.rep_overall_grade,
                    "errors": {
                        eid: {
                            "area": r.area,
                            "confidence": round(r.confidence, 3),
                            "evidence": r.evidence,
                        }
                        for eid, r in (out.rep_errors or {}).items()
                    },
                }
                reps.append(rep_data)
            elif out.rep_event and out.rep_event.incomplete:
                incompletes.append({
                    "reason": out.rep_event.incomplete_reason,
                    "at_ts_ms": ts_ms,
                })

            if raw_video_out is not None:
                annotated = frame.copy()
                _draw_deadlift_frame(annotated, out, ts_ms, error_counts,
                                     last_rep_feedback, rep_feedback_until_ms)
                if writer is None:
                    writer = _open_writer(raw_video_out, annotated.shape)
                    if writer is None:
                        print("   ! Video writer acilamadi, overlay yazilmiyor")
                        raw_video_out = None
                if writer is not None:
                    writer.write(annotated)

            frame_idx += 1
    finally:
        if writer is not None:
            writer.release()
        analyzer.close()
        cap.release()

    # Tarayici-uyumlu H.264'e transcode
    final_video_out = None
    if raw_video_out is not None:
        transcoded = _transcode_to_h264(raw_video_out)
        final_video_out = transcoded or raw_video_out

    diag["phases_seen"] = sorted(diag["phases_seen"])
    return {
        "video": video_path.name,
        "fps": fps,
        "frame_count": total,
        "rep_count": len(reps),
        "incomplete_count": len(incompletes),
        "diagnostics": diag,
        "reps": reps,
        "incompletes": incompletes,
        "error_counts": dict(error_counts),
        "overlay_video": final_video_out,
    }

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    videos = sorted(FIXTURE_DIR.glob("deadlift_*.mp4"))
    if not videos:
        print(f"Video bulunamadi: {FIXTURE_DIR}/deadlift_*.mp4")
        return
    print(f"{len(videos)} video bulundu.")
    print(f"JSON ciktilari : {OUTPUT_DIR}")
    print(f"Overlay videolar: {VIDEO_OUT_DIR}\n")
    for vp in videos:
        print(f"=> {vp.name}")
        try:
            video_out = VIDEO_OUT_DIR / f"{vp.stem}_overlay.mp4"
            result = run_video(vp, video_out_path=video_out)
            out_path = OUTPUT_DIR / f"{vp.stem}_system.json"
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            d = result["diagnostics"]
            print(f"   {result['rep_count']} rep + {result['incomplete_count']} incomplete")
            print(f"   diag: calib_frames={d['calibrating_frames']} "
                  f"calib_completed_at={d['calibration_completed_at_frame']} "
                  f"rejected={d['rejected_frames']} valid_feats={d['valid_feature_frames']}")
            if d["rejection_reasons"]:
                print(f"   rejection: {d['rejection_reasons']}")
            if result["error_counts"]:
                err_str = "  ".join(f"{k}={v}" for k, v in result["error_counts"].items())
                print(f"   errors:    {err_str}")
            print(f"   phases_seen: {d['phases_seen']}")
            if result["overlay_video"]:
                print(f"   overlay:    {Path(result['overlay_video']).relative_to(REPO_ROOT)}")
        except Exception as e:
            print(f"   HATA: {e}")
        print()
    print("Bitti.")

if __name__ == "__main__":
    main()
