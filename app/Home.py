# python -m streamlit run app/Home.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app._styles import apply_home_card_hover, apply_styles, card_icon, page_header

st.set_page_config(
    page_title="Gym Tracker",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_styles()
apply_home_card_hover()

page_header(
    title="Gym Tracker'a Hoşgeldin",
    subtitle=(
        "Webcam veya video üzerinden squat ve deadlift formunu analiz eden, "
        "tekrar sayan ve hata bildiren bir antrenör. Rutin oluştur, ilerlemeni takip et."
    ),
    eyebrow="Ana Sayfa",
)

# ---------- Navigation cards ----------

st.markdown("### Başlangıç")

c1, c2 = st.columns(2)

with c1:
    with st.container(border=True):
        card_icon("🎬")
        st.markdown("#### Video ile Analiz")
        st.markdown(
            "Çekilmiş bir egzersiz videosu yükle. İskelet, faz etiketi, tekrar sayısı "
            "ve hata tespiti overlay'li mp4 olarak döner."
        )
        st.page_link("pages/2_Video_Yukle.py", label="Video Yükle  →")

with c2:
    with st.container(border=True):
        card_icon("📹")
        st.markdown("#### Canlı Analiz")
        st.markdown(
            "Webcam ile gerçek zamanlı iskelet, faz ve tekrar takibi. "
            "Set bazlı kayıt ve rutin ilerlemesi dahil."
        )
        st.page_link("pages/1_Canli_Analiz.py", label="Canlı Analiz  →")

st.markdown("### Takip")

c3, c4 = st.columns(2)

with c3:
    with st.container(border=True):
        card_icon("📋")
        st.markdown("#### Rutinler")
        st.markdown(
            "Egzersiz programları oluştur (squat 3×10, deadlift 3×5 ...). "
            "Hedef ve gerçekleşen ilerlemeyi gör."
        )
        st.page_link("pages/3_Rutinler.py", label="Rutinler  →")

with c4:
    with st.container(border=True):
        card_icon("📊")
        st.markdown("#### Geçmiş & Dashboard")
        st.markdown(
            "Tüm oturumlar, zaman içindeki tekrar trendi, hata frekansları "
            "ve oturum detay incelemesi."
        )
        st.page_link("pages/4_Gecmis_Dashboard.py", label="Dashboard  →")

