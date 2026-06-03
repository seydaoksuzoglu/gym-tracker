import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app._styles import apply_styles, empty_state, page_header
from src.storage.repository import (
    create_routine,
    delete_routine,
    get_routine,
    get_session_detail,
    list_routines,
    list_sessions,
    mark_routine_completed,
    reopen_routine,
)


st.set_page_config(page_title="Rutinler - Gym Tracker", page_icon="📋", layout="wide")
apply_styles()
page_header(
    title="Antrenman Rutinleri",
    subtitle="Egzersiz rutinlerini oluştur, hedef ve gerçekleşen ilerlemeyi takip et.",
    eyebrow="Rutinler",
)

tab_list, tab_create = st.tabs(["Rutinlerim", "Yeni rutin"])

# ---------- Liste sekmesi ----------

with tab_list:
    routines = list_routines()
    if not routines:
        empty_state(
            icon="📋",
            title="Henüz rutin yok",
            message="'Yeni rutin' sekmesinden ilk rutinini oluşturarak başlayabilirsin.",
        )
    else:
        for r in routines:
            full = get_routine(r.id)
            if full is None:
                continue
            status = "TAMAMLANDI" if full.completed_at else "Devam ediyor"
            with st.expander(f"#{full.id} — {full.name}  ·  {status}", expanded=False):
                st.caption(f"Oluşturuldu: {full.created_at}")
                if full.completed_at:
                    st.success(f"Tamamlandı: {full.completed_at}")

                if full.items:
                    item_rows = [
                        {
                            "Sıra": it.order_index,
                            "Egzersiz": it.exercise_key,
                            "Set": it.target_sets,
                            "Tekrar": it.target_reps,
                        }
                        for it in sorted(full.items, key=lambda x: x.order_index)
                    ]
                    st.dataframe(item_rows, use_container_width=True, hide_index=True)
                    # ----- İlerleme paneli -----
                    from app._persist import compute_routine_progress
                    prog = compute_routine_progress(full.id)

                    pc1, pc2, pc3, pc4 = st.columns(4)
                    pc1.metric("Hedef set", prog["target_sets"])
                    pc2.metric("Gerçekleşen set", prog["actual_sets"])
                    pc3.metric("Hedef tekrar", prog["target_reps"])
                    pc4.metric("Gerçekleşen tekrar", prog["actual_reps"])

                    if (
                        full.completed_at is None
                        and prog["hit_target"]
                    ):
                        st.info(
                            "Hedef set sayısına ulaştın. Aşağıdaki 'Bitir' butonuyla "
                            "rutini tamamla. (Otomatik tamamlama yok.)"
                        )

                    sessions_for_routine = list_sessions(limit=200, routine_id=full.id)
                    if sessions_for_routine:
                        st.markdown("**Bu rutine ait oturumlar**")
                        sess_rows = []
                        for s in sessions_for_routine:
                            sess_rows.append({
                                "Oturum": s.id,
                                "Tarih": s.started_at.strftime("%Y-%m-%d %H:%M")
                                        if s.started_at else "-",
                                "Kaynak": s.source,
                            })
                        st.dataframe(sess_rows, use_container_width=True, hide_index=True)
                    else:
                        st.caption("Bu rutine henüz oturum bağlanmamış.")


                else:
                    st.caption("Bu rutinde egzersiz yok.")

                btn_col1, btn_col2, btn_col3 = st.columns(3)
                if full.completed_at is None:
                    if btn_col1.button("Bitir", key=f"finish_{full.id}", type="primary"):
                        mark_routine_completed(full.id)
                        st.rerun()
                else:
                    if btn_col1.button("Yeniden aç", key=f"reopen_{full.id}"):
                        reopen_routine(full.id)
                        st.rerun()

                if btn_col3.button("Sil", key=f"del_{full.id}"):
                    delete_routine(full.id)
                    st.rerun()

# ---------- Oluşturma sekmesi ----------

with tab_create:
    name = st.text_input("Rutin adı", placeholder="Alt gövde günü")

    if "new_routine_items" not in st.session_state:
        st.session_state["new_routine_items"] = []

    st.markdown("### Egzersiz ekle")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    ex = c1.selectbox("Egzersiz", ["squat", "deadlift"], key="new_item_ex")
    tset = c2.number_input("Set", min_value=1, max_value=20, value=3, key="new_item_set")
    trep = c3.number_input("Tekrar", min_value=1, max_value=50, value=10, key="new_item_rep")
    if c4.button("Ekle"):
        st.session_state["new_routine_items"].append({
            "exercise_key": ex,
            "target_sets": int(tset),
            "target_reps": int(trep),
        })

    if st.session_state["new_routine_items"]:
        st.markdown("### Eklenenler")
        st.dataframe(st.session_state["new_routine_items"], use_container_width=True, hide_index=True)
        if st.button("Listeyi temizle"):
            st.session_state["new_routine_items"] = []
            st.rerun()

    st.divider()
    save_disabled = not name or not st.session_state["new_routine_items"]
    if st.button("Rutini kaydet", type="primary", disabled=save_disabled):
        rid = create_routine(name=name, items=st.session_state["new_routine_items"])
        st.session_state["new_routine_items"] = []
        st.success(f"Rutin #{rid} kaydedildi.")
        st.rerun()
