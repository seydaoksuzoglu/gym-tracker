import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app._engine import AnalysisResult
from src.storage.repository import (
    end_session,
    save_rep,
    save_set,
    start_session,
)

def persist_analysis(
    result: AnalysisResult,
    source: str,
    video_path: Optional[str] = None,
    set_index: int = 1,
    target_reps: Optional[int] = None,
    routine_id: Optional[int] = None,
) -> dict:
    session_id = start_session(source=source, video_path=video_path, routine_id=routine_id)
    set_id = save_set(
        session_id=session_id,
        exercise_key=result.exercise,
        set_index=set_index,
        target_reps=target_reps,
        backend=result.backend,
    )
    for rep in result.reps:
        save_rep(set_id, rep.to_save_rep_dict())
    end_session(session_id)
    return {
        "session_id": session_id,
        "set_id": set_id,
        "rep_count": len(result.reps),
        "routine_id": routine_id,
    }


def compute_routine_progress(routine_id: int) -> dict:
    """Bir rutin icin hedef vs gerceklesen set/rep sayilarini hesaplar."""
    from src.storage.repository import get_routine, get_session_detail, list_sessions

    routine = get_routine(routine_id)
    if routine is None:
        return {
            "target_sets": 0, "actual_sets": 0,
            "target_reps": 0, "actual_reps": 0,
            "hit_target": False, "completed_at": None,
        }

    target_sets = sum(it.target_sets for it in routine.items)
    target_reps = sum(it.target_sets * it.target_reps for it in routine.items)

    actual_sets = 0
    actual_reps = 0
    for s in list_sessions(limit=200, routine_id=routine_id):
        detail = get_session_detail(s.id)
        if detail is None:
            continue
        for st_row in detail.sets:
            actual_sets += 1
            actual_reps += (st_row.completed_reps or 0)

    return {
        "target_sets": target_sets,
        "actual_sets": actual_sets,
        "target_reps": target_reps,
        "actual_reps": actual_reps,
        "hit_target": target_sets > 0 and actual_sets >= target_sets,
        "completed_at": routine.completed_at,
    }


