"""
UncontrolledDescentDetector birim testleri.
"""
import pytest

from src.analysis.deadlift.deadlift_features import DeadliftFeatures
from src.analysis.deadlift.error_detectors.base import RepSummary
from src.analysis.deadlift.error_detectors.uncontrolled_descent import (
    HIP_VELOCITY_THRESHOLD_PER_MS,
    RATIO_THRESHOLD,
    SUSTAINED_DROP_MS,
    UncontrolledDescentDetector,
)
from src.analysis.deadlift.phase_detector import Phase


def _feat(hip_y: float, valid: bool = True) -> DeadliftFeatures:
    return DeadliftFeatures(
        side="left", T=10.0, K=170.0, back_angle=80.0,
        hip_y=hip_y, shoulder_y=hip_y - 0.25,
        avg_visibility=0.9, valid=valid,
    )


def _summary(pull_ms: int = 1200, descent_ms: int = 1100) -> RepSummary:
    return RepSummary(
        rep_id=1,
        phase_durations_ms={
            "setup": 800, "pull": pull_ms,
            "lockout": 400, "descent": descent_ms,
        },
    )


def _feed_descent(detector, hip_series, ts_step=33, ts_start=0):
    """DESCENT fazinda verilen hip_y serisini sirayla besle."""
    for i, hip_y in enumerate(hip_series):
        detector.update(_feat(hip_y=hip_y), Phase.DESCENT,
                        ts_ms=ts_start + i * ts_step, dt_ms=ts_step)


# ---- temel davranis ----

def test_no_descent_returns_none():
    """DESCENT 0ms -> None (RepCounter incomplete dedi)."""
    d = UncontrolledDescentDetector()
    assert d.evaluate_rep(_summary(descent_ms=0)) is None


def test_controlled_descent_yields_silent_report():
    """ratio=1.0, dusuk hiz -> area=1, conf=0."""
    d = UncontrolledDescentDetector()
    # Yavas iniş: 0.30'dan 0.50'ye 1100ms'de, frame basina 0.006 hareket
    # v = 0.006 / 33 = 1.8e-4 < HIP_VELOCITY_THRESHOLD_PER_MS (4e-4) -> sinyal pasif
    _feed_descent(d, [0.30 + 0.006 * i for i in range(34)])
    rep = d.evaluate_rep(_summary(pull_ms=1100, descent_ms=1100))
    assert rep is not None
    assert rep.area == 1
    assert rep.confidence == 0.0


def test_fast_ratio_signal_a_alone():
    """ratio=0.3 -> z_a=2 -> area=3 yellow. Sinyal B silent."""
    d = UncontrolledDescentDetector()
    # Cok az frame besle -> hiz birikmez ama ratio kuçuk
    _feed_descent(d, [0.30, 0.32, 0.34])  # 3 frame, kucuk hareket
    rep = d.evaluate_rep(_summary(pull_ms=1000, descent_ms=300))
    assert rep is not None
    # ratio=0.3 -> z_a = (0.5 - 0.3) / 0.1 = 2.0 -> area=3, conf=0.5
    assert rep.area == 3
    assert rep.confidence == pytest.approx(0.5, abs=1e-6)
    assert "descent/pull=0.30" in rep.evidence


def test_sustained_drop_signal_b_alone():
    """Yuksek hiz 200ms sustained -> z_b = 200/150 ≈ 1.33 -> area=2."""
    d = UncontrolledDescentDetector()
    # Yuksek hiz: frame basina 0.02 (v = 6e-4 > 4e-4 esik)
    _feed_descent(d, [0.30 + 0.02 * i for i in range(8)])  # 8 frame = ~230ms sustained
    rep = d.evaluate_rep(_summary(pull_ms=1100, descent_ms=1100))  # ratio=1.0
    assert rep is not None
    # z_b ≈ sustained_ms/150; en az 1.0 olmali
    assert rep.area >= 2
    assert "sustained_drop=" in rep.evidence


def test_severe_drop_yields_max_area():
    """Cok uzun sustained drop -> area=5."""
    d = UncontrolledDescentDetector()
    # Cok uzun yuksek hiz: 0.02/frame x 25 frame = ~800ms
    _feed_descent(d, [0.30 + 0.02 * i for i in range(25)])
    rep = d.evaluate_rep(_summary(pull_ms=1100, descent_ms=1100))
    assert rep is not None
    assert rep.area == 5
    assert rep.confidence == 1.0


def test_or_semantics_takes_max():
    """Hem ratio=0.4 (z_a=1) hem ~330ms sustained (z_b≈2.2) -> max -> area=3."""
    d = UncontrolledDescentDetector()
    _feed_descent(d, [0.30 + 0.02 * i for i in range(12)])  # ~330ms sustained
    rep = d.evaluate_rep(_summary(pull_ms=1000, descent_ms=400))  # ratio=0.4
    assert rep is not None
    # z_a = (0.5 - 0.4) / 0.1 = 1.0
    # frame 1 prev_hip kuruyor, frame 2'den itibaren velocity var
    # sustained = (12-1)*33 - 33 = 330ms; z_b = 330/150 = 2.2
    # max = 2.2 -> area=3, conf=0.55
    assert rep.area == 3


# ---- guard'lar ----

def test_pull_phase_velocity_does_not_count():
    """PULL'da hip cabuk yukselse de DESCENT detektoru tetiklenmez."""
    d = UncontrolledDescentDetector()
    # PULL'da cok hizli
    for i in range(20):
        d.update(_feat(hip_y=0.5 - 0.02 * i), Phase.PULL,
                 ts_ms=i * 33, dt_ms=33)
    # DESCENT yavas
    _feed_descent(d, [0.3 + 0.006 * i for i in range(34)], ts_start=1000)
    rep = d.evaluate_rep(_summary(pull_ms=1100, descent_ms=1100))
    assert rep is not None
    assert rep.area == 1   # PULL'daki hiz sinyali kirletmedi


def test_invalid_features_ignored():
    d = UncontrolledDescentDetector()
    # Gecerli baslangic
    d.update(_feat(hip_y=0.30), Phase.DESCENT, ts_ms=0, dt_ms=33)
    # Invalid frame - state'i kirletmemeli
    d.update(_feat(hip_y=0.30, valid=False), Phase.DESCENT, ts_ms=33, dt_ms=33)
    rep = d.evaluate_rep(_summary())
    assert rep is not None  # ratio yine hesaplaniyor


def test_zero_pull_duration_handles_gracefully():
    """pull_ms=0 -> ratio hesaplanamaz, sinyal A None."""
    d = UncontrolledDescentDetector()
    _feed_descent(d, [0.30 + 0.006 * i for i in range(34)])
    rep = d.evaluate_rep(_summary(pull_ms=0, descent_ms=500))
    assert rep is not None
    assert "ratio n/a" in rep.evidence


# ---- reset ----

def test_reset_clears_state():
    d = UncontrolledDescentDetector()
    _feed_descent(d, [0.30 + 0.02 * i for i in range(25)])
    rep1 = d.evaluate_rep(_summary(pull_ms=1100, descent_ms=1100))
    assert rep1.area >= 3

    d.reset()
    # Reset sonrasi temiz
    rep2 = d.evaluate_rep(_summary(pull_ms=1100, descent_ms=1100))
    assert rep2 is not None
    assert rep2.area == 1


def test_thresholds_exposed():
    """Sabitler test tarafindan import edilebilir."""
    assert RATIO_THRESHOLD == 0.5
    assert SUSTAINED_DROP_MS == 150
    assert HIP_VELOCITY_THRESHOLD_PER_MS > 0
