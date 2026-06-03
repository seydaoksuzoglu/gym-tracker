import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.storage import database as db_module
from src.storage import repository
from src.storage.database import Base


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:", future=True)
    TestSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    Base.metadata.create_all(bind=engine)

    monkeypatch.setattr(db_module, "engine", engine)
    monkeypatch.setattr(db_module, "SessionLocal", TestSession)
    monkeypatch.setattr(repository, "SessionLocal", TestSession)
    yield
    engine.dispose()


def test_session_set_rep_error_roundtrip():
    session_id = repository.start_session(source="video", video_path="data/demo.mp4")
    set_id = repository.save_set(
        session_id=session_id,
        exercise_key="deadlift",
        set_index=1,
        target_reps=5,
        backend="mediapipe",
    )

    rep_dict = {
        "rep_id": 1,
        "phase_durations_ms": {"setup": 800, "pull": 1200, "lockout": 400, "descent": 1100},
        "overall_grade": "needs_attention",
        "video_ts_ms": 3500,
        "errors": {
            "uncontrolled_descent": {
                "area": 3,
                "confidence": 0.42,
                "evidence": "descent 480ms vs pull 1200ms",
            },
        },
    }
    rep_id = repository.save_rep(set_id, rep_dict)
    repository.end_session(session_id)

    detail = repository.get_session_detail(session_id)
    assert detail is not None
    assert detail.ended_at is not None
    assert len(detail.sets) == 1

    s = detail.sets[0]
    assert s.exercise_key == "deadlift"
    assert s.completed_reps == 1
    assert len(s.reps) == 1

    r = s.reps[0]
    assert r.rep_index == 1
    assert r.overall_grade == "needs_attention"
    assert r.phase_durations_ms["pull"] == 1200
    assert r.video_ts_ms == 3500
    assert len(r.errors) == 1

    e = r.errors[0]
    assert e.error_type == "uncontrolled_descent"
    assert e.area == 3
    assert e.confidence == pytest.approx(0.42)
    assert e.evidence == {"msg": "descent 480ms vs pull 1200ms"}


def test_routine_crud():
    rid = repository.create_routine(
        name="Lower body A",
        items=[
            {"exercise_key": "squat", "target_sets": 3, "target_reps": 10},
            {"exercise_key": "deadlift", "target_sets": 3, "target_reps": 5},
        ],
    )
    routine = repository.get_routine(rid)
    assert routine is not None
    assert routine.name == "Lower body A"
    assert len(routine.items) == 2
    assert routine.items[0].exercise_key == "squat"
    assert routine.items[1].target_reps == 5

    all_routines = repository.list_routines()
    assert any(r.id == rid for r in all_routines)
    assert routine.completed_at is None



def test_list_sessions_filter_by_exercise():
    s1 = repository.start_session(source="webcam")
    repository.save_set(s1, "squat", 1, 10, "mediapipe")

    s2 = repository.start_session(source="webcam")
    repository.save_set(s2, "deadlift", 1, 5, "mediapipe")

    squat_only = repository.list_sessions(exercise_key="squat")
    ids = {s.id for s in squat_only}
    assert s1 in ids
    assert s2 not in ids