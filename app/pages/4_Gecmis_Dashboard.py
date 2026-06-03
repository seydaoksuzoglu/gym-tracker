import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.storage.repository import get_session_detail, list_sessions

st.set_page_config(page_title="Gecmis - Gym Tracker", layout="wide")
st.title("Gecmis & Dashboard")

# ---------- Filtre + liste ----------

exercise_filter = st.selectbox(
    "Egzersize gore filtrele",
    ["(hepsi)", "squat", "deadlift"],
    index=0,
)
filter_key = None if exercise_filter == "(hepsi)" else exercise_filter

sessions = list_sessions(limit=200, exercise_key=filter_key)

if not sessions:
    st.info("Henuz analiz oturumu yok. Video Yukle sayfasindan baslayabilirsin.")
    st.stop()

# ---------- Toplu metrikler ----------

total_sessions = len(sessions)
total_reps = 0
exercises_seen = Counter()
errors_seen = Counter()

session_summaries = []
for s in sessions:
    detail = get_session_detail(s.id)
    if detail is None:
        continue
    session_total_reps = 0
    session_exercises = set()
    session_errors = []
    for set_row in detail.sets:
        session_exercises.add(set_row.exercise_key)
        exercises_seen[set_row.exercise_key] += 1
        for rep in set_row.reps:
            session_total_reps += 1
            for err in rep.errors:
                errors_seen[err.error_type] += 1
                session_errors.append(err.error_type)
    total_reps += session_total_reps
    session_summaries.append({
        "id": detail.id,
        "started_at": detail.started_at.strftime("%Y-%m-%d %H:%M") if detail.started_at else "-",
        "source": detail.source,
        "exercises": ", ".join(sorted(session_exercises)) or "-",
        "reps": session_total_reps,
        "errors": len(session_errors),
    })

m1, m2, m3, m4 = st.columns(4)
m1.metric("Oturum", total_sessions)
m2.metric("Toplam tekrar", total_reps)
m3.metric("En cok egzersiz", exercises_seen.most_common(1)[0][0] if exercises_seen else "-")
m4.metric("Toplam hata", sum(errors_seen.values()))

st.divider()

# ---------- Zaman serisi grafik ----------

if session_summaries:
    st.subheader("Zaman icinde rep sayisi")
    chart_dict = defaultdict(int)
    for summ in session_summaries:
        chart_dict[summ["started_at"][:10]] += summ["reps"]
    chart_data = dict(sorted(chart_dict.items()))
    if chart_data:
        st.bar_chart(chart_data)

# ---------- Hata frekansi ----------

if errors_seen:
    st.subheader("Hata frekansi")
    err_rows = [{"hata": k, "kez": v} for k, v in errors_seen.most_common()]
    st.dataframe(err_rows, use_container_width=True, hide_index=True)

# ---------- Oturum tablosu + detay ----------

st.subheader("Oturum listesi")
st.dataframe(session_summaries, use_container_width=True, hide_index=True)

selected_id = st.number_input(
    "Detay icin oturum ID gir",
    min_value=0,
    step=1,
    value=0,
)

if selected_id:
    detail = get_session_detail(int(selected_id))
    if detail is None:
        st.error(f"Oturum #{selected_id} bulunamadi.")
    else:
        st.markdown(f"### Oturum #{detail.id}")
        st.write(f"Baslangic: {detail.started_at}  ·  Bitis: {detail.ended_at or '-'}  ·  Kaynak: {detail.source}")
        if detail.video_path:
            st.caption(f"Video: `{detail.video_path}`")

        for set_row in detail.sets:
            st.markdown(f"**Set #{set_row.set_index} — {set_row.exercise_key} ({set_row.backend})**")
            rep_rows = []
            for rep in set_row.reps:
                rep_rows.append({
                    "rep": rep.rep_index,
                    "grade": rep.overall_grade or "-",
                    "video_ts_ms": rep.video_ts_ms,
                    "phases_ms": rep.phase_durations_ms,
                    "errors": ", ".join(e.error_type for e in rep.errors) or "-",
                })
            if rep_rows:
                st.dataframe(rep_rows, use_container_width=True, hide_index=True)
            else:
                st.caption("Bu set'te kayitli rep yok.")
