# python -m streamlit run app/Home.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

st.set_page_config(page_title="Gym Tracker", layout="wide")
st.title("Gym Tracker")
st.write("Poz kestirimi tabanli form analizi ve antrenman takibi.")

col1, col2 = st.columns(2)
with col1:
    exercise = st.radio(
        "Egzersiz",
        ["squat", "deadlift"],
        horizontal=True,
        key="exercise",
    )
with col2:
    source = st.radio(
        "Giris kaynagi",
        ["video", "canli"],
        horizontal=True,
        key="source",
        help="Video yukle veya webcam'den canli analiz",
    )

st.divider()

if source == "video":
    st.info("Sol menudeki **Video Yukle** sayfasindan devam et.")
elif source == "canli":
    st.warning("Canli analiz Sprint D'de devreye girecek. Simdilik video yukle.")
