import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app._engine import analyze_video
from app._persist import compute_routine_progress, persist_analysis
from app._styles import apply_styles, page_header
from src.storage.repository import list_routines, mark_routine_completed


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Video Yukle - Gym Tracker", page_icon="🎬", layout="wide")
apply_styles()
page_header(
    title="Video ile Analiz",
    subtitle="Çekmiş olduğun bir egzersiz videosunu yükle. İskelet, faz, tekrar ve hata overlay'li mp4 olarak döner.",
    eyebrow="Yüklenen Video",
)

exercise = st.radio(
    "Egzersiz",
    ["squat", "deadlift"],
    horizontal=True,
    key="exercise",
)

_routines = [r for r in list_routines() if r.completed_at is None]
_routine_options: list[tuple[str, int | None]] = [("(bağımsız oturum)", None)]
for _r in _routines:
    _routine_options.append((f"#{_r.id} — {_r.name}", _r.id))

routine_choice = st.selectbox(
    "Rutin (opsiyonel)",
    options=_routine_options,
    format_func=lambda x: x[0],
    index=0,
    help="Bu video bir rutinin parçasıysa seçin; bağımsız analiz için 'bağımsız oturum' seçin.",
)
selected_routine_id = routine_choice[1]


uploaded = st.file_uploader(
    "Egzersiz videosu",
    type=["mp4", "mov", "avi"],
    help="Yan kameradan çekilmiş 5-30 saniyelik klip önerilir (deadlift için ilk ~1 sn dik durulmalı — kalibrasyon).",
)

if uploaded is not None:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = str(OUTPUTS_DIR / f"analysis_{exercise}_{timestamp}.mp4")

    st.caption(f"Giriş: `{tmp_path}`  ·  Çıktı: `{output_video_path}`")

    if st.button("Analizi başlat", type="primary"):
        with st.spinner("Analiz ediliyor — uzun videoda 30-90 saniye sürebilir"):
            try:
                result = analyze_video(
                    path=tmp_path,
                    exercise=exercise,
                    output_video_path=output_video_path,
                )
            except Exception as e:
                st.error("Analiz hatası")
                st.exception(e)
                st.stop()

        try:
            persist_info = persist_analysis(
                result=result,
                source="video",
                video_path=uploaded.name,
                routine_id=selected_routine_id,
            )
            st.session_state["last_persist"] = persist_info
        except Exception as e:
            st.warning(f"Veritabanına kayıt hatası: {e}")
            st.session_state["last_persist"] = None

        st.session_state["last_result"] = result


# ----- Sonuç render (re-run sonrası da kalır) -----

result = st.session_state.get("last_result")
if result is not None:
    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Toplam tekrar", result.total_reps)
    m2.metric("İşlenen kare", result.total_frames)
    m3.metric("Reddedilen kare", result.rejected_frames)
    m4.metric("Arka uç", result.backend)

    for note in result.notes:
        st.warning(note)

    if result.output_video_path and Path(result.output_video_path).exists():
        st.subheader("Üzerine işlenmiş video")
        try:
            with open(result.output_video_path, "rb") as f:
                video_bytes = f.read()
            col_l, col_v, col_r = st.columns([1, 1, 1])
            with col_v:
                st.video(video_bytes)
        except Exception:
            st.warning("Tarayıcıda oynatma başarısız olabilir, dosyayı indirip yerel oynatıcıda aç.")
        st.download_button(
            "Üzerine işlenmiş videoyu indir",
            data=video_bytes,
            file_name=Path(result.output_video_path).name,
            mime="video/mp4",
        )


    if not result.reps:
        st.warning("Tamamlanan tekrar yakalanamadı. Kamera açısı / kalibrasyon durumunu kontrol et.")
    else:
        st.subheader("Tekrar dökümü")

        # Hata özeti
        error_summary: dict[str, dict] = {}
        for rep in result.reps:
            for err_type, payload in rep.errors.items():
                bucket = error_summary.setdefault(err_type, {"count": 0, "conf_sum": 0.0})
                bucket["count"] += 1
                bucket["conf_sum"] += payload.get("confidence", 0.0)

        if error_summary:
            err_rows = [
                {
                    "Hata": k,
                    "Kaç tekrar": v["count"],
                    "Ort. güven": round(v["conf_sum"] / max(v["count"], 1), 2),
                }
                for k, v in error_summary.items()
            ]
            st.markdown("**Hata frekansı**")
            st.dataframe(err_rows, use_container_width=True, hide_index=True)

        # Per-rep tablo
        st.markdown("**Tekrar detayları**")
        table_rows = []
        for rep in result.reps:
            phases = rep.phase_durations_ms or {}
            table_rows.append({
                "Tekrar": rep.rep_index,
                "Not": rep.overall_grade or "-",
                "Video ms": rep.video_ts_ms,
                "Hazırlık ms": phases.get("setup"),
                "Çekiş ms": phases.get("pull"),
                "Kilit ms": phases.get("lockout"),
                "İniş ms": phases.get("descent"),
                "Hatalar": ", ".join(rep.errors.keys()) if rep.errors else "-",
            })
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        # Faz süreleri bar chart (deadlift'te dolu, squat'ta boş kalır)
        phase_has_data = any(r.phase_durations_ms for r in result.reps)
        if phase_has_data:
            st.markdown("**Faz süreleri (ms) — tekrar bazlı**")
            chart_data = {
                "Hazırlık": [(r.phase_durations_ms or {}).get("setup", 0) for r in result.reps],
                "Çekiş": [(r.phase_durations_ms or {}).get("pull", 0) for r in result.reps],
                "Kilit": [(r.phase_durations_ms or {}).get("lockout", 0) for r in result.reps],
                "İniş": [(r.phase_durations_ms or {}).get("descent", 0) for r in result.reps],
            }
            st.bar_chart(chart_data)

        # Rep bazlı detaylı expander
        st.markdown("**Hata detayları (tekrar bazlı)**")
        for rep in result.reps:
            if not rep.errors:
                continue
            with st.expander(f"Tekrar {rep.rep_index} — {rep.overall_grade or 'not yok'}"):
                for err_type, payload in rep.errors.items():
                    st.write(f"**{err_type}**  ·  bölge={payload.get('area')}  ·  güven={payload.get('confidence'):.2f}")
                    ev = payload.get("evidence")
                    if ev:
                        st.code(str(ev), language="text")

        # Ham JSON
        with st.expander("Ham analiz çıktısı (geliştirici)"):
            st.json({
                "exercise": result.exercise,
                "backend": result.backend,
                "total_frames": result.total_frames,
                "rejected_frames": result.rejected_frames,
                "reps": [r.to_save_rep_dict() for r in result.reps],
            })

    persist_info = st.session_state.get("last_persist")
    if persist_info:
        st.success(
            f"Veritabanına yazıldı  ·  oturum #{persist_info['session_id']}  ·  "
            f"set #{persist_info['set_id']}  ·  {persist_info['rep_count']} tekrar"
        )
        routine_id_for_panel = persist_info.get("routine_id")
        if routine_id_for_panel:
            prog = compute_routine_progress(routine_id_for_panel)
            st.markdown(f"### Rutin ilerlemesi (#{routine_id_for_panel})")
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Hedef set", prog["target_sets"])
            pc2.metric("Gerçekleşen set", prog["actual_sets"])
            pc3.metric("Hedef tekrar", prog["target_reps"])
            pc4.metric("Gerçekleşen tekrar", prog["actual_reps"])

            if prog["completed_at"]:
                st.success(f"Bu rutin zaten tamamlandı: {prog['completed_at']}")
            elif prog["hit_target"]:
                st.info("Hedef set sayısına ulaştın. Aşağıdaki butonla rutini bitirebilirsin.")
                if st.button("Rutini bitir", type="primary", key=f"finish_{routine_id_for_panel}"):
                    mark_routine_completed(routine_id_for_panel)
                    st.success("Rutin tamamlandı olarak işaretlendi.")
                    st.rerun()


