"""Ortak UI stilleri.

Tema: Koyu sidebar + temiz beyaz content + emerald accent.
Brand sidebar'a CSS pseudo-element ile inject ediliyor (DOM bagimsiz).
"""
import streamlit as st


_C = {
    "side_bg":       "#0F172A",
    "side_bg_alt":   "#111B2E",
    "side_text":     "#CBD5E1",
    "side_text_mid": "#94A3B8",
    "side_border":   "#1E293B",
    "side_hover":    "#1E2A44",

    "bg":            "#F7F8FA",
    "card":          "#FFFFFF",
    "border":        "#E5E7EB",
    "border_soft":   "#EEF0F3",
    "text":          "#0F172A",
    "text_soft":     "#475569",
    "muted":         "#94A3B8",

    "accent":        "#10B981",
    "accent_2":      "#14B8A6",
    "accent_dark":   "#059669",
    "accent_soft":   "#D1FAE5",
    "warning":       "#F59E0B",
    "danger":        "#EF4444",
}


_CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: {_C['text']};
    }}
    h1, h2, h3, h4 {{
        font-family: 'Sora', 'Inter', sans-serif;
        letter-spacing: -0.022em;
        font-weight: 700;
        color: {_C['text']};
    }}
    h1 {{ font-size: 2.4rem !important; line-height: 1.1; }}
    h2 {{ font-size: 1.6rem !important; }}
    h3 {{ font-size: 1.15rem !important; }}

    .stApp {{
        background: {_C['bg']} !important;
    }}

    /* Streamlit'in ust header'i default beyaz — sayfa zeminine kaynasin */
    header[data-testid="stHeader"] {{
        background: transparent !important;
        backdrop-filter: none !important;
    }}

    section.main > div.block-container,
    div[data-testid="stMainBlockContainer"] {{
        padding-top: 1.2rem !important;
        padding-bottom: 4rem !important;
        max-width: 1240px;
    }}

    /* ===================================================== */
    /* SIDEBAR — koyu                                          */
    /* ===================================================== */
    section[data-testid="stSidebar"] {{
        background: {_C['side_bg']} !important;
        border-right: 1px solid {_C['side_border']} !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: {_C['side_text']};
    }}

    /* ===================================================== */
    /* SIDEBAR BRAND — title outer ::before, subtitle inner ::before */
    /* ===================================================== */
    section[data-testid="stSidebar"] > div:first-child::before {{
        content: '🏋️  Gym Tracker';
        display: block;
        padding: 1.4rem 1.2rem 0.2rem 1.2rem;
        background: {_C['side_bg']};
        color: #F1F5F9;
        font-family: 'Sora', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }}
    section[data-testid="stSidebar"] > div:first-child > div:first-child::before {{
        content: 'Form analizi ve rutin takibi';
        display: block;
        padding: 0 1.2rem 1.1rem 1.2rem;
        margin: -0.2rem -1rem 0.6rem -1rem;
        background: {_C['side_bg']};
        color: {_C['side_text_mid']};
        font-size: 0.74rem;
        border-bottom: 1px solid {_C['side_border']};
    }}

    /* Sidebar headings (Ayarlar, Oturum gibi) */
    section[data-testid="stSidebar"] h3 {{
        font-size: 0.72rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        color: {_C['side_text_mid']} !important;
        font-weight: 700 !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: {_C['side_border']} !important;
        margin: 0.7rem 0 !important;
    }}

    /* Collapse butonunu Streamlit default davranisina birak —
       sidebar kapali iken sol uste erisilebilir, acik iken sidebar header'inda. */

    /* Page nav */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{
        color: {_C['side_text']} !important;
        border-radius: 10px !important;
        padding: 0.5rem 0.8rem !important;
        transition: background 0.12s ease, color 0.12s ease !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {{
        background: {_C['side_hover']} !important;
        color: #FFFFFF !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {{
        background: linear-gradient(135deg, {_C['accent']} 0%, {_C['accent_2']} 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }}

    section[data-testid="stSidebar"] button[kind="primary"] {{
        background: linear-gradient(135deg, {_C['accent']} 0%, {_C['accent_2']} 100%) !important;
        color: #FFFFFF !important;
        border: 0 !important;
    }}
    section[data-testid="stSidebar"] button:not([kind="primary"]):not([data-testid="stSidebarCollapseButton"]) {{
        background: {_C['side_bg_alt']} !important;
        border: 1px solid {_C['side_border']} !important;
        color: {_C['side_text']} !important;
    }}
    section[data-testid="stSidebar"] button:not([kind="primary"]):not([data-testid="stSidebarCollapseButton"]):hover {{
        background: {_C['side_hover']} !important;
        border-color: {_C['accent']} !important;
        color: #FFFFFF !important;
    }}
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] div[data-baseweb="input"] > div {{
        background: {_C['side_bg_alt']} !important;
        border-color: {_C['side_border']} !important;
        color: {_C['side_text']} !important;
    }}

    /* ===================================================== */
    /* CONTENT                                                  */
    /* ===================================================== */

    button[kind="primary"] {{
        background: linear-gradient(135deg, {_C['accent']} 0%, {_C['accent_2']} 100%) !important;
        border: 0 !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.3rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(16,185,129,0.25) !important;
    }}
    button[kind="primary"]:hover {{
        transform: translateY(-1px);
        box-shadow: 0 8px 22px rgba(16,185,129,0.32) !important;
    }}

    div.stButton > button:not([kind="primary"]) {{
        background: {_C['card']} !important;
        border: 1px solid {_C['border']} !important;
        color: {_C['text']} !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.1rem !important;
        font-weight: 500 !important;
    }}
    div.stButton > button:not([kind="primary"]):hover {{
        border-color: {_C['accent']} !important;
        color: {_C['accent_dark']} !important;
    }}

    a[data-testid="stPageLink"] {{
        display: inline-flex !important;
        align-items: center;
        gap: 0.4rem;
        padding: 0.55rem 1rem !important;
        margin-top: 0.6rem;
        background: {_C['card']} !important;
        border: 1px solid {_C['border']} !important;
        border-radius: 10px !important;
        color: {_C['text']} !important;
        font-weight: 600 !important;
        text-decoration: none !important;
        transition: all 0.15s ease !important;
    }}
    a[data-testid="stPageLink"]:hover {{
        background: linear-gradient(135deg, {_C['accent']} 0%, {_C['accent_2']} 100%) !important;
        color: #FFFFFF !important;
        border-color: transparent !important;
        transform: translateX(2px);
    }}

    /* ===================================================== */
    /* CARDS — sade gorunum (tum bordered container'lar)         */
    /* ===================================================== */
    section.main div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {_C['card']} !important;
        border: 1px solid {_C['border_soft']} !important;
        border-radius: 18px !important;
        padding: 0.8rem 1rem !important;
        box-shadow: 0 1px 3px rgba(15,23,42,0.04),
                    0 4px 16px rgba(15,23,42,0.04) !important;
    }}

    /* Card icon block (Home nav kartlarinda kullanilir) */
    .gt-card-icon {{
        width: 54px;
        height: 54px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, {_C['accent_soft']} 0%, #ECFDF5 100%);
        border: 1px solid {_C['border']};
        border-radius: 14px;
        font-size: 1.7rem;
        margin-bottom: 0.8rem;
        transition: transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1),
                    background 0.3s ease,
                    box-shadow 0.3s ease;
    }}

    /* Metric cards */
    div[data-testid="stMetric"] {{
        background: {_C['card']};
        border: 1px solid {_C['border_soft']};
        border-radius: 14px;
        padding: 1rem 1.1rem;
        position: relative;
        overflow: hidden;
    }}
    div[data-testid="stMetric"]::before {{
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        background: linear-gradient(180deg, {_C['accent']} 0%, {_C['accent_2']} 100%);
    }}
    div[data-testid="stMetricLabel"] p {{
        font-size: 0.7rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: {_C['muted']} !important;
        font-weight: 600 !important;
    }}
    div[data-testid="stMetricValue"] {{
        font-family: 'Sora', sans-serif !important;
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: {_C['text']} !important;
        line-height: 1.1 !important;
    }}

    button[data-baseweb="tab"] {{
        font-weight: 600 !important;
        color: {_C['muted']} !important;
        padding: 0.6rem 0.2rem !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {_C['accent_dark']} !important;
    }}
    div[data-baseweb="tab-highlight"] {{
        background: linear-gradient(90deg, {_C['accent']}, {_C['accent_2']}) !important;
        height: 3px !important;
        border-radius: 2px !important;
    }}

    section.main div[data-baseweb="select"] > div,
    section.main div[data-baseweb="input"] > div {{
        border-radius: 10px !important;
        border-color: {_C['border']} !important;
    }}

    div[data-testid="stFileUploader"] section {{
        border: 2px dashed {_C['border']} !important;
        border-radius: 14px !important;
        background: {_C['bg']} !important;
    }}

    div[data-testid="stDataFrame"] {{
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid {_C['border_soft']};
    }}

    div[data-testid="stAlert"] {{
        border-radius: 12px !important;
        border: 0 !important;
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    hr {{
        border-color: {_C['border_soft']} !important;
        margin: 2rem 0 !important;
    }}

    div[data-testid="stCaptionContainer"] {{
        color: {_C['muted']} !important;
    }}

    /* ===================================================== */
    /* PAGE HEADER + EMPTY                                      */
    /* ===================================================== */
    .gt-page-header {{
        position: relative;
        padding: 1.6rem 1.8rem 1.4rem 1.8rem;
        margin: 0 0 1.8rem 0;
        background: {_C['card']};
        border: 1px solid {_C['border_soft']};
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(15,23,42,0.04);
    }}
    .gt-page-header::before {{
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 5px;
        background: linear-gradient(180deg, {_C['accent']} 0%, {_C['accent_2']} 100%);
    }}
    .gt-page-header .gt-eyebrow {{
        display: inline-block;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {_C['accent_dark']};
        font-weight: 700;
        background: {_C['accent_soft']};
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        margin-bottom: 0.8rem;
    }}
    .gt-page-header h1 {{
        margin: 0 0 0.4rem 0 !important;
        color: {_C['text']};
    }}
    .gt-page-header p {{
        color: {_C['text_soft']};
        margin: 0;
        font-size: 1rem;
        max-width: 720px;
    }}

    .gt-empty {{
        text-align: center;
        padding: 3.5rem 1.5rem;
        background: {_C['card']};
        border: 1px dashed {_C['border']};
        border-radius: 18px;
    }}
    .gt-empty .gt-empty-icon {{
        font-size: 3rem;
        margin-bottom: 0.6rem;
    }}
    .gt-empty h3 {{
        margin: 0.2rem 0 0.5rem 0 !important;
        color: {_C['text']};
    }}
    .gt-empty p {{
        color: {_C['text_soft']};
        margin: 0 auto;
        max-width: 460px;
    }}
</style>
"""


def apply_styles() -> None:
    """Tüm sayfalarda ortak CSS uygula. Brand pseudo-element ile sidebar'a inject olur."""
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


_HOME_CARD_HOVER_CSS = f"""
<style>
    /* Sadece Home'da uygulanir — :has() kullanmadan, tum bordered container'a
       hover ekler. Home'da bordered container yalnizca nav kartlaridir. */
    section.main div[data-testid="stVerticalBlockBorderWrapper"] {{
        transition: transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1),
                    box-shadow 0.35s ease,
                    border-color 0.35s ease !important;
        cursor: pointer !important;
        will-change: transform !important;
    }}
    section.main div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
        transform: translateY(-12px) scale(1.015) !important;
        border-color: {_C['accent']} !important;
        box-shadow: 0 10px 24px rgba(15,23,42,0.08),
                    0 40px 80px rgba(16,185,129,0.32) !important;
    }}
    section.main div[data-testid="stVerticalBlockBorderWrapper"]:hover h3,
    section.main div[data-testid="stVerticalBlockBorderWrapper"]:hover h4 {{
        color: {_C['accent_dark']} !important;
        transition: color 0.3s ease !important;
    }}
    section.main div[data-testid="stVerticalBlockBorderWrapper"]:hover .gt-card-icon {{
        transform: rotate(-10deg) scale(1.15) !important;
        background: linear-gradient(135deg, {_C['accent']} 0%, {_C['accent_2']} 100%) !important;
        box-shadow: 0 10px 24px rgba(16,185,129,0.45) !important;
    }}
    section.main div[data-testid="stVerticalBlockBorderWrapper"]:hover a[data-testid="stPageLink"] {{
        background: linear-gradient(135deg, {_C['accent']} 0%, {_C['accent_2']} 100%) !important;
        color: #FFFFFF !important;
        border-color: transparent !important;
    }}
</style>
"""


def apply_home_card_hover() -> None:
    """Yalnizca Home.py'de cagrilir — nav kartlarina hover efekti ekler."""
    st.markdown(_HOME_CARD_HOVER_CSS, unsafe_allow_html=True)


def card_icon(emoji: str) -> None:
    st.markdown(f'<div class="gt-card-icon">{emoji}</div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    eyebrow_html = f'<div class="gt-eyebrow">{eyebrow}</div>' if eyebrow else ""
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="gt-page-header">{eyebrow_html}'
        f"<h1>{title}</h1>{subtitle_html}</div>",
        unsafe_allow_html=True,
    )


def empty_state(icon: str, title: str, message: str) -> None:
    st.markdown(
        f'<div class="gt-empty">'
        f'<div class="gt-empty-icon">{icon}</div>'
        f"<h3>{title}</h3><p>{message}</p></div>",
        unsafe_allow_html=True,
    )
