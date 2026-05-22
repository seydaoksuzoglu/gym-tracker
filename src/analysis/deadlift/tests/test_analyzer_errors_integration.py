"""
Integration: iki detector + scoring beraber. Analyzer'in yaptigi glue mantigi.
LiveDeadliftAnalyzer'i mock'lamadan, dedektor + summary uzerinde calisir.
"""
from src.analysis.deadlift.deadlift_features import DeadliftFeatures
from src.analysis.deadlift.error_detectors.base import RepSummary
from src.analysis.deadlift.error_detectors.incomplete_lockout import (
    IncompleteLockoutDetector,
)
from src.analysis.deadlift.error_detectors.uncontrolled_descent import (
    UncontrolledDescentDetector,
)
from src.analysis.deadlift.phase_detector import Phase
from src.analysis.deadlift.scoring import overall_grade


def _feat(T=10.0, hip_y=0.30) -> DeadliftFeatures:
    return DeadliftFeatures(
        side="left", T=T, K=170.0, back_angle=90 - T,
        hip_y=hip_y, shoulder_y=hip_y - 0.25,
        avg_visibility=0.9, valid=True,
    )


def _evaluate_all(detectors, summary):
    """Analyzer'in yaptigi: tum dedektorleri cagir, errors dict olustur."""
    errors = {}
    for d in detectors:
        r = d.evaluate_rep(summary)
        if r is not None:
            errors[r.error_id] = r
    grade = overall_grade([r.area for r in errors.values()])
    return errors, grade


def test_perfect_rep_yields_perfect_grade():
    """Mukemmel rep: LOCKOUT T_min=0, ratio=1.0, dusuk hiz -> grade=perfect."""
    lockout_det = IncompleteLockoutDetector()
    descent_det = UncontrolledDescentDetector()

    # LOCKOUT phase: T sifira yakin
    for i, T in enumerate([3.0, 2.0, 1.0, 0.5]):
        lockout_det.update(_feat(T=T), Phase.LOCKOUT, ts_ms=i * 33, dt_ms=33)

    # DESCENT phase: yavas iniş
    for i in range(34):
        descent_det.update(_feat(T=10.0, hip_y=0.30 + 0.006 * i),
                           Phase.DESCENT, ts_ms=1000 + i * 33, dt_ms=33)

    summary = RepSummary(rep_id=1, phase_durations_ms={
        "setup": 800, "pull": 1100, "lockout": 400, "descent": 1100,
    })
    errors, grade = _evaluate_all([lockout_det, descent_det], summary)

    assert len(errors) == 2  # ikisi de rapor verir (area=1)
    assert errors["incomplete_lockout"].area == 1
    assert errors["uncontrolled_descent"].area == 1
    assert grade == "perfect"


def test_bad_rep_takes_worst_area():
    """Bir hata area=4, digeri area=1 -> grade=form_issue."""
    lockout_det = IncompleteLockoutDetector()
    descent_det = UncontrolledDescentDetector()

    # LOCKOUT'ta T_min=9.5 -> pseudo_z=3 -> area=4
    for i, T in enumerate([10.0, 9.8, 9.5]):
        lockout_det.update(_feat(T=T), Phase.LOCKOUT, ts_ms=i * 33, dt_ms=33)

    # DESCENT iyi (kontrollu)
    for i in range(34):
        descent_det.update(_feat(T=10.0, hip_y=0.30 + 0.006 * i),
                           Phase.DESCENT, ts_ms=1000 + i * 33, dt_ms=33)

    summary = RepSummary(rep_id=1, phase_durations_ms={
        "setup": 800, "pull": 1100, "lockout": 400, "descent": 1100,
    })
    errors, grade = _evaluate_all([lockout_det, descent_det], summary)

    assert errors["incomplete_lockout"].area == 4
    assert errors["uncontrolled_descent"].area == 1
    assert grade == "form_issue"


def test_incomplete_rep_no_lockout_filters_detector():
    """LOCKOUT hic yoksa incomplete_lockout dedektoru None doner."""
    lockout_det = IncompleteLockoutDetector()
    descent_det = UncontrolledDescentDetector()

    # PULL ve DESCENT, ama LOCKOUT yok
    for i in range(34):
        descent_det.update(_feat(hip_y=0.30 + 0.006 * i),
                           Phase.DESCENT, ts_ms=i * 33, dt_ms=33)

    summary = RepSummary(rep_id=1, phase_durations_ms={
        "setup": 800, "pull": 1100, "lockout": 0, "descent": 1100,
    })
    errors, grade = _evaluate_all([lockout_det, descent_det], summary)

    # incomplete_lockout None dondu, dict'te yok
    assert "incomplete_lockout" not in errors
    # uncontrolled_descent var (DESCENT gozlendi)
    assert "uncontrolled_descent" in errors
