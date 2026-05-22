"""
RepCounter birim testleri.
Sentetik PhaseEvent dizilerinden rep sayimini dogrular.
"""
from src.analysis.deadlift.phase_detector import Phase, PhaseEvent
from src.analysis.deadlift.rep_counter import RepCounter


def _ev(phase: Phase, ts: int, from_phase: Phase = None, changed: bool = True) -> PhaseEvent:
    return PhaseEvent(
        phase=phase, phase_changed=changed, ts_ms=ts, from_phase=from_phase,
    )


def test_initial_state_is_zero():
    rc = RepCounter()
    assert rc.rep_count == 0
    assert rc.incomplete_count == 0


def test_full_cycle_counts_one_rep():
    rc = RepCounter()
    rc.update(_ev(Phase.SETUP, ts=0, changed=False))
    rc.update(_ev(Phase.PULL, ts=500, from_phase=Phase.SETUP))
    rc.update(_ev(Phase.LOCKOUT, ts=1500, from_phase=Phase.PULL))
    rc.update(_ev(Phase.DESCENT, ts=1900, from_phase=Phase.LOCKOUT))
    out = rc.update(_ev(Phase.SETUP, ts=3000, from_phase=Phase.DESCENT))

    assert out.rep_completed is True
    assert out.rep_id == 1
    assert rc.rep_count == 1
    assert rc.incomplete_count == 0


def test_missing_lockout_marks_incomplete():
    rc = RepCounter()
    rc.update(_ev(Phase.SETUP, ts=0, changed=False))
    rc.update(_ev(Phase.PULL, ts=500, from_phase=Phase.SETUP))
    # LOCKOUT atlandi
    rc.update(_ev(Phase.DESCENT, ts=1400, from_phase=Phase.PULL))
    out = rc.update(_ev(Phase.SETUP, ts=2500, from_phase=Phase.DESCENT))

    assert out.rep_completed is False
    assert out.incomplete is True
    assert out.incomplete_reason == "no_lockout"
    assert rc.rep_count == 0
    assert rc.incomplete_count == 1


def test_consecutive_reps_via_descent_to_pull():
    """Ardisik rep'ler: DESCENT'tan SETUP'a ugramadan PULL'a donus."""
    rc = RepCounter()
    rc.update(_ev(Phase.PULL, ts=0, changed=False))
    rc.update(_ev(Phase.LOCKOUT, ts=1000, from_phase=Phase.PULL))
    rc.update(_ev(Phase.DESCENT, ts=1400, from_phase=Phase.LOCKOUT))
    # Direkt PULL'a (SETUP'a ugramadan) - cycle kapanir
    out1 = rc.update(_ev(Phase.PULL, ts=2500, from_phase=Phase.DESCENT))
    assert out1.rep_completed is True
    assert rc.rep_count == 1

    # 2. rep
    rc.update(_ev(Phase.LOCKOUT, ts=3500, from_phase=Phase.PULL))
    rc.update(_ev(Phase.DESCENT, ts=3900, from_phase=Phase.LOCKOUT))
    out2 = rc.update(_ev(Phase.PULL, ts=5000, from_phase=Phase.DESCENT))
    assert out2.rep_completed is True
    assert rc.rep_count == 2


def test_phase_durations_accumulated():
    """phase_durations_ms her fazda gecirilen toplam sureyi yansitmali."""
    rc = RepCounter()
    rc.update(_ev(Phase.SETUP, ts=0, changed=False))
    rc.update(_ev(Phase.PULL, ts=1000, from_phase=Phase.SETUP))
    rc.update(_ev(Phase.LOCKOUT, ts=2000, from_phase=Phase.PULL))
    rc.update(_ev(Phase.DESCENT, ts=2500, from_phase=Phase.LOCKOUT))
    out = rc.update(_ev(Phase.SETUP, ts=3500, from_phase=Phase.DESCENT))

    assert out.rep_completed is True
    d = out.phase_durations_ms
    # Suresinin sirasi: setup 0->1000=1000, pull 1000->2000=1000,
    # lockout 2000->2500=500, descent 2500->3500=1000
    assert d["setup"] == 1000
    assert d["pull"] == 1000
    assert d["lockout"] == 500
    assert d["descent"] == 1000


def test_no_event_emitted_when_no_phase_change():
    """phase_changed=False event'lerinde RepEvent bos donmeli."""
    rc = RepCounter()
    out = rc.update(_ev(Phase.SETUP, ts=100, changed=False))
    assert out.rep_completed is False
    assert out.incomplete is False
