import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

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


st.set_page_config(page_title="Rutinler - Gym Tracker", layout="wide")
st.title("Antrenman Rutinleri")

tab_list, tab_create = st.tabs(["Rutinlerim", "Yeni rutin"])

# ---------- Liste sekmesi ----------

with tab_list:
    routines = list_routines()
    if not routines:
        st.info("Henuz rutin yok. 'Yeni rutin' sekmesinden ekleyebilirsin.")
    else:
        for r in routines:
            full = get_routine(r.id)
            if full is None:
                continue
            status = "TAMAMLANDI" if full.completed_at else "Devam"
            with st.expander(f"#{full.id} — {full.name}  ·  {status}", expanded=False):
                st.caption(f"Olusturuldu: {full.created_at}")
                if full.completed_at:
                    st.success(f"Tamamlandi: {full.completed_at}")

                if full.items:
                    item_rows = [
                        {
                            "sira": it.order_index,
                            "egzersiz": it.exercise_key,
                            "set": it.target_sets,
                            "rep": it.target_reps,
                        }
                        for it in sorted(full.items, key=lambda x: x.order_index)
                    ]
                    st.dataframe(item_rows, use_container_width=True, hide_index=True)
                    # ----- Ilerleme paneli -----
                    from app._persist import compute_routine_progress
                    prog = compute_routine_progress(full.id)

                    pc1, pc2, pc3, pc4 = st.columns(4)
                    pc1.metric("Hedef set", prog["target_sets"])
                    pc2.metric("Gercek set", prog["actual_sets"])
                    pc3.metric("Hedef rep", prog["target_reps"])
                    pc4.metric("Gercek rep", prog["actual_reps"])

                    if (
                        full.completed_at is None
                        and prog["hit_target"]
                    ):
                        st.info(
                            "Hedef set sayisina ulastin. Asagidaki 'Bitir' butonuyla "
                            "rutini tamamla. (Otomatik tamamlama yok.)"
                        )

                    sessions_for_routine = list_sessions(limit=200, routine_id=full.id)
                    if sessions_for_routine:
                        st.markdown("**Bu rutine ait oturumlar**")
                        sess_rows = []
                        for s in sessions_for_routine:
                            sess_rows.append({
                                "oturum": s.id,
                                "tarih": s.started_at.strftime("%Y-%m-%d %H:%M")
                                        if s.started_at else "-",
                                "kaynak": s.source,
                            })
                        st.dataframe(sess_rows, use_container_width=True, hide_index=True)
                    else:
                        st.caption("Bu rutine henuz oturum baglanmamis.")


                else:
                    st.caption("Bu rutinde egzersiz yok.")

                btn_col1, btn_col2, btn_col3 = st.columns(3)
                if full.completed_at is None:
                    if btn_col1.button("Bitir", key=f"finish_{full.id}", type="primary"):
                        mark_routine_completed(full.id)
                        st.rerun()
                else:
                    if btn_col1.button("Yeniden ac", key=f"reopen_{full.id}"):
                        reopen_routine(full.id)
                        st.rerun()

                if btn_col3.button("Sil", key=f"del_{full.id}"):
                    delete_routine(full.id)
                    st.rerun()

# ---------- Oluşturma sekmesi ----------

with tab_create:
    name = st.text_input("Rutin adi", placeholder="Lower body A")

    if "new_routine_items" not in st.session_state:
        st.session_state["new_routine_items"] = []

    st.markdown("### Egzersiz ekle")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    ex = c1.selectbox("Egzersiz", ["squat", "deadlift"], key="new_item_ex")
    tset = c2.number_input("Set", min_value=1, max_value=20, value=3, key="new_item_set")
    trep = c3.number_input("Rep", min_value=1, max_value=50, value=10, key="new_item_rep")
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
