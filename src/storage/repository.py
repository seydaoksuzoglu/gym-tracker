from contextlib import contextmanager
from datetime import datetime
from typing import Iterator, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession, selectinload

from src.storage.database import SessionLocal
from src.storage.models import (
    Rep,
    RepError,
    Routine,
    RoutineItem,
    Session as SessionRow,
    SetRow,
)


@contextmanager
def session_scope() -> Iterator[OrmSession]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------- Sessions ----------

def start_session(
    source: str,
    video_path: Optional[str] = None,
    routine_id: Optional[int] = None,
) -> int:
    with session_scope() as db:
        row = SessionRow(source=source, video_path=video_path, routine_id=routine_id)
        db.add(row)
        db.flush()
        return row.id


def end_session(session_id: int) -> None:
    with session_scope() as db:
        row = db.get(SessionRow, session_id)
        if row is None:
            raise ValueError(f"Session {session_id} not found")
        row.ended_at = datetime.now()


# ---------- Sets ----------

def save_set(
    session_id: int,
    exercise_key: str,
    set_index: int,
    target_reps: Optional[int],
    backend: str,
) -> int:
    with session_scope() as db:
        row = SetRow(
            session_id=session_id,
            exercise_key=exercise_key,
            set_index=set_index,
            target_reps=target_reps,
            backend=backend,
            started_at=datetime.now(),
        )
        db.add(row)
        db.flush()
        return row.id


# ---------- Reps ----------

def save_rep(set_id: int, rep_dict: dict) -> int:
    """rep_dict = analizorun urettigi JSON (bkz deadlift CLAUDE.md ornegi)."""
    with session_scope() as db:
        rep = Rep(
            set_id=set_id,
            rep_index=rep_dict["rep_id"],
            overall_grade=rep_dict.get("overall_grade"),
            phase_durations_ms=rep_dict.get("phase_durations_ms"),
            video_ts_ms=rep_dict.get("video_ts_ms"),
        )
        db.add(rep)
        db.flush()

        for err_type, err_data in (rep_dict.get("errors") or {}).items():
            evidence = err_data.get("evidence")
            if isinstance(evidence, str):
                evidence = {"msg": evidence}
            db.add(RepError(
                rep_id=rep.id,
                error_type=err_type,
                area=err_data["area"],
                confidence=err_data["confidence"],
                evidence=evidence,
            ))

        # Incomplete rep'ler kayda alinir ama set.completed_reps'i artirmaz
        # (rutin hedefi sadece tam rep'leri saysin).
        is_incomplete = rep_dict.get("overall_grade") == "incomplete"
        set_row = db.get(SetRow, set_id)
        if set_row is not None and not is_incomplete:
            set_row.completed_reps = (set_row.completed_reps or 0) + 1

        return rep.id


# ---------- Queries ----------

def list_sessions(
    limit: int = 50,
    exercise_key: Optional[str] = None,
    routine_id: Optional[int] = None,
) -> list[SessionRow]:
    with session_scope() as db:
        stmt = select(SessionRow).order_by(SessionRow.started_at.desc()).limit(limit)
        if exercise_key:
            stmt = stmt.join(SetRow).where(SetRow.exercise_key == exercise_key).distinct()
        if routine_id is not None:
            stmt = stmt.where(SessionRow.routine_id == routine_id)
        return list(db.execute(stmt).scalars())


def get_session_detail(session_id: int) -> Optional[SessionRow]:
    with session_scope() as db:
        stmt = (
            select(SessionRow)
            .where(SessionRow.id == session_id)
            .options(
                selectinload(SessionRow.sets)
                .selectinload(SetRow.reps)
                .selectinload(Rep.errors)
            )
        )
        return db.execute(stmt).scalar_one_or_none()


# ---------- Routines ----------

def create_routine(name: str, items: list[dict]) -> int:
    """items: [{"exercise_key": "squat", "target_sets": 3, "target_reps": 10}]"""
    with session_scope() as db:
        routine = Routine(name=name)
        db.add(routine)
        db.flush()
        for idx, it in enumerate(items):
            db.add(RoutineItem(
                routine_id=routine.id,
                exercise_key=it["exercise_key"],
                target_sets=it["target_sets"],
                target_reps=it["target_reps"],
                order_index=it.get("order_index", idx),
            ))
        return routine.id


def list_routines() -> list[Routine]:
    with session_scope() as db:
        stmt = select(Routine).order_by(Routine.created_at.desc())
        return list(db.execute(stmt).scalars())


def get_routine(routine_id: int) -> Optional[Routine]:
    with session_scope() as db:
        stmt = (
            select(Routine)
            .where(Routine.id == routine_id)
            .options(selectinload(Routine.items))
        )
        return db.execute(stmt).scalar_one_or_none()
    
def mark_routine_completed(routine_id: int) -> None:
    with session_scope() as db:
        row = db.get(Routine, routine_id)
        if row is None:
            raise ValueError(f"Routine {routine_id} not found")
        row.completed_at = datetime.now()


def reopen_routine(routine_id: int) -> None:
    with session_scope() as db:
        row = db.get(Routine, routine_id)
        if row is None:
            raise ValueError(f"Routine {routine_id} not found")
        row.completed_at = None


def delete_routine(routine_id: int) -> None:
    with session_scope() as db:
        row = db.get(Routine, routine_id)
        if row is None:
            return
        db.delete(row)

