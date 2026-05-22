"""
IncompleteLockoutDetector birim testleri.
"""
import pytest

from src.analysis.deadlift.deadlift_features import DeadliftFeatures
from src.analysis.deadlift.error_detectors.base import RepSummary
from src.analysis.deadlift.error_detectors.incomplete_lockout import (
    IncompleteLockoutDetector,
    T_LOCKOUT_THRESHOLD,
)
from src.analysis.deadlift.phase_detector import Phase


def _feat(T: float, valid: bool = True) -> DeadliftFeatures:
    return DeadliftFeatures(
        side="left", T=T, K=178.0, back_angle=90 - T,
        hip_y=0.30, shoulder_y=0.05, avg_visibility=0.9, valid=valid,
    )


def _summary() -> RepSummary:
    return RepSummary(
        rep_id=1,
        phase_durations_ms={"setup": 800, "pull": 1200, "lockout": 400, "descent": 1100},
    )


def _feed_lockout(detector, T_values, ts_step=33):
    """LOCKOUT phase'inde verilen T degerlerini sirayla besle."""
    for i, T in enumerate(T_values):
        detector.update(_feat(T=T), Phase.LOCKOUT, ts_ms=i * ts_step, dt_ms=ts_step)


# ---- temel davranis ----

def test_no_lockout_observed_returns_none():
    """LOCKOUT hic gozlenmediyse evaluate_rep None doner."""
    d = IncompleteLockoutDetector()
    # Sadece PULL frame'leri besle
    for i, T in enumerate([35, 25, 15]):
        d.update(_feat(T=T), Phase.PULL, ts_ms=i * 33, dt_ms=33)
    assert d.evaluate_rep(_summary()) is None


def test_perfect_lockout_t_zero_returns_silent_report():
    """T_min=0 -> area=1, confidence=0 (silent ama rapor var)."""
    d = IncompleteLockoutDetector()
    _feed_lockout(d, [2.0, 1.0, 0.5, 0.3, 0.2])
    rep = d.evaluate_rep(_summary())
    assert rep is not None
    assert rep.error_id == "incomplete_lockout"
    assert rep.area == 1
    assert rep.confidence == 0.0
    assert "within" in rep.evidence


def test_t_min_exactly_at_threshold_is_silent():
    """T_min=5 -> pseudo_z=0 -> area=1."""
    d = IncompleteLockoutDetector()
    _feed_lockout(d, [6.0, 5.5, 5.0])
    rep = d.evaluate_rep(_summary())
    assert rep is not None
    assert rep.area == 1
    assert rep.confidence == 0.0


def test_t_min_slightly_over_yields_yellow():
    """T_min=8 -> pseudo_z=2 -> area=3, conf=0.5 (yellow)."""
    d = IncompleteLockoutDetector()
    _feed_lockout(d, [9.0, 8.5, 8.0, 8.2])
    rep = d.evaluate_rep(_summary())
    assert rep is not None
    assert rep.area == 3
    assert rep.confidence == pytest.approx(0.5, abs=1e-6)
    assert "> 5.0" in rep.evidence


def test_t_min_severe_yields_max_area():
    """T_min=12 -> pseudo_z=4.67 -> area=5, conf=1.0 (capped)."""
    d = IncompleteLockoutDetector()
    _feed_lockout(d, [15.0, 13.0, 12.0])
    rep = d.evaluate_rep(_summary())
    assert rep is not None
    assert rep.area == 5
    assert rep.confidence == 1.0


# ---- guard'lar ----

def test_pull_phase_T_below_threshold_does_not_count():
    """PULL'da T<5 olusa bile LOCKOUT izlenmiyor."""
    d = IncompleteLockoutDetector()
    # PULL'da T=2 (mantiksal degil ama guard testi)
    d.update(_feat(T=2.0), Phase.PULL, ts_ms=0, dt_ms=33)
    # Sonra LOCKOUT'ta T_min=8
    _feed_lockout(d, [8.0, 8.5, 8.0])
    rep = d.evaluate_rep(_summary())
    assert rep is not None
    # T_min PULL'daki 2 degil, LOCKOUT'taki 8
    assert "T_min=8.0" in rep.evidence


def test_invalid_features_ignored():
    d = IncompleteLockoutDetector()
    d.update(_feat(T=0.0, valid=False), Phase.LOCKOUT, ts_ms=0, dt_ms=33)
    # Hala None - LOCKOUT'ta valid frame gormedi
    assert d.evaluate_rep(_summary()) is None


# ---- reset ----

def test_reset_clears_state():
    d = IncompleteLockoutDetector()
    _feed_lockout(d, [8.0, 8.5])
    assert d.evaluate_rep(_summary()) is not None
    d.reset()
    # Reset sonrasi LOCKOUT gormeyen taze detektor gibi
    assert d.evaluate_rep(_summary()) is None


def test_threshold_constant_exposed():
    """T_LOCKOUT_THRESHOLD modul seviyesinde, test edilebilir."""
    assert T_LOCKOUT_THRESHOLD == 5.0
