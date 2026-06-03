import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import av
import cv2
import mediapipe as mp
import streamlit as st
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from src.config import settings
from src.vis.skeleton_drawer import (
    draw_landmarks_on_image_mediapipe,
    draw_phase_label,
    draw_rep_counter,
)

from app._persist import compute_routine_progress, persist_set
from app._styles import apply_styles, page_header
from src.storage.repository import (
    end_session,
    list_routines,
    mark_routine_completed,
    start_session,
)


st.set_page_config(page_title="Canli Analiz - Gym Tracker", page_icon="📹", layout="wide")
apply_styles()
page_header(
    title="Canlı Analiz",
    subtitle="Webcam'den canlı iskelet, faz, tekrar ve hata bildirimi. Rutin seçip set bazlı veritabanına yazabilirsin.",
    eyebrow="Webcam Akışı",
)

# ---------- Sidebar ayarlar + oturum ----------

if "live_session_id" not in st.session_state:
    st.session_state["live_session_id"] = None
if "live_set_index" not in st.session_state:
    st.session_state["live_set_index"] = 1
if "live_routine_id" not in st.session_state:
    st.session_state["live_routine_id"] = None
if "just_completed_routine" not in st.session_state:
    st.session_state["just_completed_routine"] = None

# ---------- Rutin tamamlanma kutlamasi (re-run sonrasi) ----------

_completed = st.session_state.get("just_completed_routine")
if _completed:
    st.balloons()
    st.success(
        f"🎉  Rutin tamamlandı: **{_completed['name']}**  "
        f"(hedef {_completed['target_sets']} set × {_completed['target_reps']} tekrar)"
    )
    st.session_state["just_completed_routine"] = None

with st.sidebar:
    st.markdown("### Ayarlar")
    exercise = st.radio(
        "Egzersiz",
        ["squat", "deadlift"],
        key="live_exercise",
        horizontal=True,
        help="Değiştirmek için önce oturumu kapat, seç, yeniden aç.",
    )

    st.markdown("---")
    st.markdown("### Oturum")

    if st.session_state["live_session_id"] is None:
        _routines = [r for r in list_routines() if r.completed_at is None]
        _opts: list[tuple[str, int | None]] = [("(bağımsız oturum)", None)]
        for _r in _routines:
            _opts.append((f"#{_r.id} — {_r.name}", _r.id))
        _choice = st.selectbox(
            "Rutin (opsiyonel)",
            options=_opts,
            format_func=lambda x: x[0],
            index=0,
        )
        if st.button("Oturumu başlat", type="primary"):
            sid = start_session(source="webcam", routine_id=_choice[1])
            st.session_state["live_session_id"] = sid
            st.session_state["live_routine_id"] = _choice[1]
            st.session_state["live_set_index"] = 1
            st.rerun()
    else:
        st.success(f"Aktif oturum #{st.session_state['live_session_id']}")
        if st.session_state["live_routine_id"]:
            st.caption(f"Rutin #{st.session_state['live_routine_id']}")
        if st.button("Oturumu kapat"):
            end_session(st.session_state["live_session_id"])
            st.session_state["live_session_id"] = None
            st.session_state["live_routine_id"] = None
            st.session_state["live_set_index"] = 1
            st.rerun()

    st.markdown("---")
    st.caption("Arka uç: MediaPipe.")


def _make_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=str(settings.mediapipe_model_abs))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


_DEADLIFT_CONNS = [(11, 23), (12, 24), (23, 25), (24, 26),
                   (25, 27), (26, 28), (11, 12), (23, 24)]
_GRADE_COLORS = {
    "perfect": (0, 255, 0),
    "good": (100, 220, 100),
    "needs_attention": (0, 200, 255),
    "form_issue": (0, 100, 255),
    "severe": (0, 0, 255),
}


def _draw_box(img, lines, x, y, color, font_scale=0.7, thickness=2, line_gap=28):
    if not lines:
        return img
    max_w = 0
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        max_w = max(max_w, w)
    total_h = len(lines) * line_gap + 10
    cv2.rectangle(img, (x - 10, y - 28), (x + max_w + 20, y - 28 + total_h), (0, 0, 0), -1)
    yy = y
    for line in lines:
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)
        yy += line_gap
    return img


# ---------- Deadlift processor ----------

class DeadliftLiveProcessor(VideoProcessorBase):
    def __init__(self):
        from src.analysis.deadlift.deadlift_analyzer import LiveDeadliftAnalyzer
        self.analyzer = LiveDeadliftAnalyzer(model_path=str(settings.mediapipe_model_abs))
        self._frame_idx = 0
        self._error_counts = {
            "incomplete_lockout": 0,
            "uncontrolled_descent": 0,
            "incomplete_rep": 0,
        }
        self._last_rep_feedback = None
        self._rep_feedback_until_ms = 0
        self.completed_reps: list[dict] = []
        self._attempt_counter = 0
        self.reset_requested = False

    def _do_reset(self):
        from src.analysis.deadlift.deadlift_analyzer import LiveDeadliftAnalyzer
        try:
            self.analyzer.close()
        except Exception:
            pass
        self.analyzer = LiveDeadliftAnalyzer(model_path=str(settings.mediapipe_model_abs))
        self.completed_reps.clear()
        self._attempt_counter = 0
        self._error_counts = {
            "incomplete_lockout": 0,
            "uncontrolled_descent": 0,
            "incomplete_rep": 0,
        }
        self._last_rep_feedback = None
        self._rep_feedback_until_ms = 0
        self._frame_idx = 0

    def recv(self, frame):
        if self.reset_requested:
            self._do_reset()
            self.reset_requested = False

        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ts_ms = self._frame_idx * 33
        self._frame_idx += 1

        analysis = self.analyzer.analyze(img_rgb, ts_ms)
        annotated = img_bgr.copy()

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
            samples = len(self.analyzer.calibrator._torso_samples)
            required = self.analyzer.calibrator.frames_required
            if self.analyzer.calibrator.has_failed():
                cv2.putText(annotated, "KALIBRASYON BASARISIZ",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(annotated, "Stop > Start ile yeniden dene, sabit dur",
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(annotated, f"CALIBRATING {samples}/{required}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                if samples < 5:
                    cv2.putText(annotated, "Tum vucut kamerada gorunsun (dizler dahil)",
                                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                elif samples >= required:
                    cv2.putText(annotated, "Tam sabit dur, son kontrol...",
                                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        elif analysis.rejected_reason:
            cv2.putText(annotated, f"REJECTED: {analysis.rejected_reason}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            draw_phase_label(annotated, analysis.phase, ts_ms)
            draw_rep_counter(annotated, analysis.rep_count, analysis.incomplete_count)
            err_txt = (f"ILO:{self._error_counts['incomplete_lockout']}  "
                       f"UCD:{self._error_counts['uncontrolled_descent']}  "
                       f"INC:{self._error_counts['incomplete_rep']}")
            (tw, th), _ = cv2.getTextSize(err_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            x = annotated.shape[1] - tw - 20
            y = 40
            cv2.rectangle(annotated, (x - 10, y - th - 10),
                          (x + tw + 10, y + 10), (0, 0, 0), -1)
            cv2.putText(annotated, err_txt, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

        if analysis.rep_event and analysis.rep_event.rep_completed:
            flagged: list[str] = []
            errors_out: dict[str, dict] = {}
            for eid, report in (analysis.rep_errors or {}).items():
                errors_out[eid] = {
                    "area": report.area,
                    "confidence": report.confidence,
                    "evidence": getattr(report, "evidence", None),
                }
                if report.confidence >= 0.05:
                    self._error_counts[eid] = self._error_counts.get(eid, 0) + 1
                    flagged.append(f"{eid} c={report.confidence:.2f}")

            grade = analysis.rep_overall_grade or "?"
            self._last_rep_feedback = {
                "rep_id": analysis.rep_event.rep_id,
                "grade": grade,
                "flagged": flagged,
                "color": _GRADE_COLORS.get(grade, (255, 255, 255)),
            }
            self._rep_feedback_until_ms = ts_ms + 1500

            self._attempt_counter += 1
            self.completed_reps.append({
                "rep_id": self._attempt_counter,
                "overall_grade": analysis.rep_overall_grade,
                "phase_durations_ms": analysis.rep_event.phase_durations_ms,
                "video_ts_ms": ts_ms,
                "errors": errors_out,
            })

        elif analysis.rep_event and analysis.rep_event.incomplete:
            self._error_counts["incomplete_rep"] += 1
            reason_text = {
                "no_lockout": "lockout'a ulasilmadi (tam dik durulmadi)",
                "missing_phase": "faz dizilimi eksik",
            }.get(analysis.rep_event.incomplete_reason or "", "bilinmeyen sebep")
            self._last_rep_feedback = {
                "rep_id": None,
                "grade": "INCOMPLETE",
                "flagged": [reason_text],
                "color": (0, 165, 255),
            }
            self._rep_feedback_until_ms = ts_ms + 2000

            self._attempt_counter += 1
            self.completed_reps.append({
                "rep_id": self._attempt_counter,
                "overall_grade": "incomplete",
                "phase_durations_ms": analysis.rep_event.phase_durations_ms,
                "video_ts_ms": ts_ms,
                "errors": {
                    "incomplete_rep": {
                        "area": 0,
                        "confidence": 1.0,
                        "evidence": reason_text,
                    }
                },
            })

        if self._last_rep_feedback and ts_ms < self._rep_feedback_until_ms:
            if self._last_rep_feedback["rep_id"] is None:
                first_line = self._last_rep_feedback["grade"]
            else:
                first_line = (f"REP {self._last_rep_feedback['rep_id']}: "
                              f"{self._last_rep_feedback['grade'].upper()}")
            lines = [first_line]
            for eid in self._last_rep_feedback["flagged"]:
                lines.append(f"  - {eid}")
            y0 = 160
            for i, line in enumerate(lines):
                cv2.putText(annotated, line, (10, y0 + i * 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            self._last_rep_feedback["color"], 2)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ---------- Squat processor ----------

class SquatLiveProcessor(VideoProcessorBase):
    def __init__(self):
        from src.analysis.squat.squat_analyzer import LiveSquatAnalyzer
        self.landmarker = _make_landmarker()
        self.analyzer = LiveSquatAnalyzer()
        self._frame_idx = 0
        self._last_live_warnings: list = []
        self._live_warning_until_ms = 0
        self._last_rep_feedback = None
        self._rep_feedback_until_ms = 0
        self.completed_reps: list[dict] = []
        self.reset_requested = False

    def _do_reset(self):
        from src.analysis.squat.squat_analyzer import LiveSquatAnalyzer
        self.analyzer = LiveSquatAnalyzer()
        self.completed_reps.clear()
        self._last_live_warnings = []
        self._live_warning_until_ms = 0
        self._last_rep_feedback = None
        self._rep_feedback_until_ms = 0
        self._frame_idx = 0

    def recv(self, frame):
        if self.reset_requested:
            self._do_reset()
            self.reset_requested = False

        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ts_ms = self._frame_idx * 33
        self._frame_idx += 1

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        pose_result = self.landmarker.detect_for_video(mp_image, ts_ms)
        analysis = self.analyzer.analyze(pose_result, ts_ms)

        if getattr(analysis, "state", None) == "descent":
            self._last_rep_feedback = None
            self._rep_feedback_until_ms = 0
        if getattr(analysis, "live_warnings", None):
            self._last_live_warnings = analysis.live_warnings
            self._live_warning_until_ms = ts_ms + 1200
        if getattr(analysis, "rep_feedback", None) is not None:
            self._last_rep_feedback = analysis.rep_feedback
            self._rep_feedback_until_ms = ts_ms + 700

        current_count = self.analyzer.counter.rep_count
        if current_count > len(self.completed_reps):
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
            self.completed_reps.append({
                "rep_id": current_count,
                "overall_grade": grade,
                "phase_durations_ms": None,
                "video_ts_ms": ts_ms,
                "errors": errors_out,
            })

        annotated_rgb = draw_landmarks_on_image_mediapipe(img_rgb, pose_result)
        annotated = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        if ts_ms < self._live_warning_until_ms and self._last_live_warnings:
            annotated = _draw_box(
                annotated, [f"UYARI: {w}" for w in self._last_live_warnings],
                x=10, y=40, color=(0, 165, 255),
            )

        if ts_ms < self._rep_feedback_until_ms and self._last_rep_feedback is not None:
            if self._last_rep_feedback.has_error:
                lines = [f"REP {self._last_rep_feedback.rep_count} HATALI"]
                lines += [f"HATA: {lab}" for lab in self._last_rep_feedback.error_labels]
                color = (0, 0, 255)
            else:
                lines = [f"REP {self._last_rep_feedback.rep_count} DOGRU FORM"]
                color = (0, 255, 0)
            annotated = _draw_box(annotated, lines, x=10, y=160, color=color)

        cv2.putText(annotated, f"REP: {self.analyzer.counter.rep_count}",
                    (10, annotated.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ---------- Factory ----------

def _processor_factory():
    if exercise == "deadlift":
        return DeadliftLiveProcessor()
    return SquatLiveProcessor()


with st.container(border=True):
    st.markdown(f"#### Kamera akışı — {exercise}")
    col_l, col_v, col_r = st.columns([1, 2, 1])
    with col_v:
        ctx = webrtc_streamer(
            key=f"pose-live-{exercise}",
            video_processor_factory=_processor_factory,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

st.info(
    "Egzersizi değiştirmek için önce oturumu kapat, seç, yeniden aç. "
    "Deadlift için ilk ~1 sn kalibrasyon amacıyla dik durulmalı."
)

# ---------- Set kontrol ----------

if (
    st.session_state["live_session_id"] is not None
    and ctx is not None
    and ctx.video_processor is not None
):
    proc = ctx.video_processor
    pending_count = len(proc.completed_reps)

    st.markdown("### Aktif set")
    m1, m2, m3 = st.columns(3)
    m1.metric("Set #", st.session_state["live_set_index"])
    m2.metric("Bekleyen tekrar", pending_count)
    m3.metric("Egzersiz", exercise)

    st.caption(
        "Not: 'Bekleyen tekrar' sayısı sayfa render'ında donmuş kalır. "
        "Aşağıdaki 'Sayımı yenile' ile güncelle; 'Set'i bitir' tıklamasında zaten en son sayım okunur."
    )
    bc1, bc2, bc3 = st.columns(3)
    if bc1.button("Set'i bitir ve kaydet", type="primary"):
        fresh_reps = list(proc.completed_reps)
        if not fresh_reps:
            st.warning("Bekleyen tekrar bulunamadı. Önce egzersiz yap, sonra 'Sayımı yenile'ye bas.")
        else:
            info = persist_set(
                session_id=st.session_state["live_session_id"],
                exercise=exercise,
                backend="mediapipe",
                set_index=st.session_state["live_set_index"],
                reps=fresh_reps,
            )
            st.session_state["live_set_index"] += 1
            proc.reset_requested = True
            st.success(f"Set #{info['set_id']} kaydedildi · {info['rep_count']} tekrar")
            st.rerun()

    if bc2.button("Set'i iptal et (sayacı sıfırla)"):
        proc.reset_requested = True
        st.rerun()

    if bc3.button("Sayımı yenile"):
        st.rerun()

    # Rutin ilerlemesi
    rid = st.session_state.get("live_routine_id")
    if rid:
        prog = compute_routine_progress(rid)
        st.markdown(f"#### Rutin #{rid} ilerlemesi")
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Hedef set", prog["target_sets"])
        pc2.metric("Gerçekleşen set", prog["actual_sets"])
        pc3.metric("Hedef tekrar", prog["target_reps"])
        pc4.metric("Gerçekleşen tekrar", prog["actual_reps"])
        if prog["completed_at"]:
            st.success(f"Rutin tamamlandı: {prog['completed_at']}")
        elif prog["hit_target"]:
            st.info("Hedef set sayısına ulaştın. Aşağıdaki butonla rutini bitirebilirsin.")
            if st.button("Rutini bitir", type="primary", key=f"finish_live_{rid}"):
                from src.storage.repository import get_routine
                _routine = get_routine(rid)
                mark_routine_completed(rid)
                st.session_state["just_completed_routine"] = {
                    "id": rid,
                    "name": _routine.name if _routine else f"#{rid}",
                    "target_sets": prog["target_sets"],
                    "target_reps": prog["target_reps"],
                }
                st.rerun()
