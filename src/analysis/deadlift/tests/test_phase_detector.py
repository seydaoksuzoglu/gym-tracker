"""
PhaseDetector birim testleri.
Sentetik DeadliftFeatures dizileriyle her faz gecisini test eder.
Gercek video gerekmez - deterministik.
"""
from src.analysis.deadlift.calibration import StandingBaseline
from src.analysis.deadlift.deadlift_features import DeadliftFeatures
from src.analysis.deadlift.phase_detector import PhaseDetector, Phase


def _feat(T: float, hip_y: float, valid: bool = True) -> DeadliftFeatures:
    return DeadliftFeatures(
        side="left", T=T, K=170.0, back_angle=90 - T,
        hip_y=hip_y, shoulder_y=hip_y - 0.25,
        avg_visibility=0.9, valid=valid,
    )


def _baseline(standing_hip_y: float = 0.45) -> StandingBaseline:
    return StandingBaseline(
        torso_length=0.25,
        shoulder_width=0.04,
        femur_length=0.20,
        standing_hip_y=standing_hip_y,
    )


def test_initial_phase_is_setup():
    det = PhaseDetector(baseline=_baseline())
    assert det.phase == Phase.SETUP


def test_invalid_feature_does_not_change_phase():
    det = PhaseDetector(baseline=_baseline())
    ev = det.update(_feat(T=0, hip_y=0, valid=False), ts_ms=0)
    assert ev.phase == Phase.SETUP
    assert ev.phase_changed is False


def test_pull_triggered_when_hip_rises():
    """SETUP -> PULL: hip yukseliyor (y azaliyor), MIN_PHASE_HOLD_MS sonra commit."""
    det = PhaseDetector(baseline=_baseline(standing_hip_y=0.45))
    hip_y = 0.60
    last_ev = None
    for ts in range(0, 300, 33):
        hip_y -= 0.005  # 5e-3/33 = ~1.5e-4 -> VELOCITY_EPS=5e-5'in uzerinde
        last_ev = det.update(_feat(T=40, hip_y=hip_y), ts_ms=ts)
    assert det.phase == Phase.PULL


def test_lockout_requires_300ms_hold():
    """PULL -> LOCKOUT: T<8 + |v|<eps, en az 300ms tutulmali."""
    det = PhaseDetector(baseline=_baseline())

    # Once PULL'a gec
    hip_y = 0.55
    for ts in range(0, 300, 33):
        hip_y -= 0.005
        det.update(_feat(T=20, hip_y=hip_y), ts_ms=ts)
    assert det.phase == Phase.PULL

    # 200ms sabit dur (lockout candidate ama hold yetmez)
    for ts in range(300, 500, 33):
        det.update(_feat(T=3, hip_y=0.45), ts_ms=ts)
    assert det.phase == Phase.PULL  # henuz commit etmedi

    # 300ms+ daha tut -> commit
    for ts in range(500, 900, 33):
        det.update(_feat(T=3, hip_y=0.45), ts_ms=ts)
    assert det.phase == Phase.LOCKOUT


def test_lockout_to_descent_when_hip_falls():
    det = PhaseDetector(baseline=_baseline())

    # PULL'a gec
    hip_y = 0.55
    for ts in range(0, 300, 33):
        hip_y -= 0.005
        det.update(_feat(T=20, hip_y=hip_y), ts_ms=ts)

    # LOCKOUT'a gec
    for ts in range(300, 800, 33):
        det.update(_feat(T=3, hip_y=0.45), ts_ms=ts)
    assert det.phase == Phase.LOCKOUT

    # DESCENT: hip iniyor
    hip_y = 0.45
    for ts in range(800, 1200, 33):
        hip_y += 0.005
        det.update(_feat(T=30, hip_y=hip_y), ts_ms=ts)
    assert det.phase == Phase.DESCENT


def test_single_frame_anomaly_does_not_trigger_lockout():
    """Tek frame'lik T=3 anomalisi LOCKOUT'a gecirmemeli (multi-frame onayi)."""
    det = PhaseDetector(baseline=_baseline())

    hip_y = 0.55
    for ts in range(0, 300, 33):
        hip_y -= 0.005
        det.update(_feat(T=20, hip_y=hip_y), ts_ms=ts)
    assert det.phase == Phase.PULL

    # Tek frame T=3 (parazit), sonra geri PULL
    det.update(_feat(T=3, hip_y=hip_y), ts_ms=333)
    det.update(_feat(T=20, hip_y=hip_y - 0.005), ts_ms=366)
    assert det.phase == Phase.PULL  # gecmedi


def test_from_phase_filled_on_transition():
    """phase_changed=True olan event from_phase'i dolu olmali."""
    det = PhaseDetector(baseline=_baseline())
    hip_y = 0.60
    transition_event = None
    for ts in range(0, 500, 33):
        hip_y -= 0.005
        ev = det.update(_feat(T=40, hip_y=hip_y), ts_ms=ts)
        if ev.phase_changed:
            transition_event = ev
            break
    assert transition_event is not None
    assert transition_event.from_phase == Phase.SETUP
    assert transition_event.phase == Phase.PULL
