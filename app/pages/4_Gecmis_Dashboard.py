import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app._styles import apply_styles, empty_state, page_header
from src.storage.repository import get_session_detail, list_sessions

ERROR_CONFIDENCE_THRESHOLD = 0.3

st.set_page_config(page_title="Gecmis - Gym Tracker", page_icon="📊", layout="wide")
apply_styles()
page_header(
    title="Geçmiş & Dashboard",
    subtitle="Tüm oturumlar, zaman içindeki tekrar sayısı, hata frekansları ve oturum detayları.",
    eyebrow="Analiz",
)

# ---------- Filtre + liste ----------

exercise_filter = st.selectbox(
    "Egzersize göre filtrele",
    ["(hepsi)", "squat", "deadlift"],
    index=0,
)
filter_key = None if exercise_filter == "(hepsi)" else exercise_filter

sessions = list_sessions(limit=200, exercise_key=filter_key)

if not sessions:
    empty_state(
        icon="📊",
        title="Henüz analiz oturumu yok",
        message="Video Yükle veya Canlı Analiz sayfasından ilk oturumunu oluşturabilirsin.",
    )
    st.stop()

# ---------- Toplu metrikler ----------

total_sessions = len(sessions)
total_reps = 0
exercises_seen = Counter()
errors_seen = Counter()

session_summaries = []
# (gun, egzersiz) -> toplam rep sayisi (set x rep)
by_day_exercise: dict[tuple[str, str], int] = defaultdict(int)

for s in sessions:
    detail = get_session_detail(s.id)
    if detail is None:
        continue
    day_str = detail.started_at.strftime("%Y-%m-%d") if detail.started_at else None
    session_total_reps = 0
    session_exercises = set()
    session_errors = []
    for set_row in detail.sets:
        session_exercises.add(set_row.exercise_key)
        exercises_seen[set_row.exercise_key] += 1
        set_rep_count = 0
        for rep in set_row.reps:
            session_total_reps += 1
            set_rep_count += 1
            for err in rep.errors:
                if err.confidence < ERROR_CONFIDENCE_THRESHOLD:
                    continue
                errors_seen[err.error_type] += 1
                session_errors.append(err.error_type)
        if day_str and set_rep_count > 0:
            by_day_exercise[(day_str, set_row.exercise_key)] += set_rep_count
    total_reps += session_total_reps
    session_summaries.append({
        "Oturum": detail.id,
        "Başlangıç": detail.started_at.strftime("%Y-%m-%d %H:%M") if detail.started_at else "-",
        "Kaynak": detail.source,
        "Egzersizler": ", ".join(sorted(session_exercises)) or "-",
        "Tekrar": session_total_reps,
        "Hata": len(session_errors),
    })

m1, m2, m3, m4 = st.columns(4)
m1.metric("Oturum", total_sessions)
m2.metric("Toplam tekrar", total_reps)
m3.metric("En çok egzersiz", exercises_seen.most_common(1)[0][0] if exercises_seen else "-")
m4.metric("Toplam hata", sum(errors_seen.values()))

st.divider()

# ---------- Gun x Egzersiz rep grafigi ----------

if by_day_exercise:
    st.subheader("Egzersize göre günlük tekrar sayısı")
    st.caption("Her bar, o gün ilgili egzersizde tamamlanan toplam tekrar sayısıdır (set × rep).")
    days_sorted = sorted({d for d, _ in by_day_exercise.keys()})
    exercises_sorted = sorted({e for _, e in by_day_exercise.keys()})
    chart_df = pd.DataFrame(
        {
            ex: [by_day_exercise.get((day, ex), 0) for day in days_sorted]
            for ex in exercises_sorted
        },
        index=days_sorted,
    )
    chart_df.index.name = "Tarih"
    st.bar_chart(chart_df)

# ---------- Hata frekansı ----------

if errors_seen:
    st.subheader("Hata frekansı")
    err_rows = [{"Hata": k, "Kez": v} for k, v in errors_seen.most_common()]
    st.dataframe(err_rows, use_container_width=True, hide_index=True)

# ---------- Oturum tablosu + detay ----------

st.subheader("Oturum listesi")
st.dataframe(session_summaries, use_container_width=True, hide_index=True)

selected_id = st.number_input(
    "Detay için oturum ID gir",
    min_value=0,
    step=1,
    value=0,
)

if selected_id:
    detail = get_session_detail(int(selected_id))
    if detail is None:
        st.error(f"Oturum #{selected_id} bulunamadı.")
    else:
        st.markdown(f"### Oturum #{detail.id}")
        st.write(f"Başlangıç: {detail.started_at}  ·  Bitiş: {detail.ended_at or '-'}  ·  Kaynak: {detail.source}")
        if detail.video_path:
            st.caption(f"Video: `{detail.video_path}`")

        for set_row in detail.sets:
            st.markdown(f"**Set #{set_row.set_index} — {set_row.exercise_key} ({set_row.backend})**")
            rep_rows = []
            for rep in set_row.reps:
                visible_errors = [
                    f"{e.error_type} (güven={e.confidence:.2f})"
                    for e in rep.errors
                    if e.confidence >= ERROR_CONFIDENCE_THRESHOLD
                ]
                ts_ms = rep.video_ts_ms
                ts_label = f"{ts_ms/1000:.1f}s" if ts_ms else "-"
                phases = rep.phase_durations_ms or {}
                phase_label = (
                    " · ".join(f"{k}={v}ms" for k, v in phases.items())
                    if phases else "-"
                )
                rep_rows.append({
                    "Tekrar": rep.rep_index,
                    "Not": rep.overall_grade or "-",
                    "Zaman": ts_label,
                    "Faz süreleri": phase_label,
                    "Hatalar": ", ".join(visible_errors) or "-",
                })
            if rep_rows:
                st.dataframe(rep_rows, use_container_width=True, hide_index=True)
            else:
                st.caption("Bu set'te kayıtlı tekrar yok.")
