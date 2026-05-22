"""
ErrorDetector ABC + MultiFrameConfirmation birim testleri.
"""
from typing import Optional

import pytest

from src.analysis.deadlift.deadlift_features import DeadliftFeatures
from src.analysis.deadlift.error_detectors.base import (
    ErrorDetector,
    ErrorReport,
    MultiFrameConfirmation,
    RepSummary,
)
from src.analysis.deadlift.phase_detector import Phase


# ---- ErrorDetector ABC ----

class _DummyRepDetector(ErrorDetector):
    """Rep-level dedektor ornegi: phase durations'a bakar."""
    error_id = "dummy_rep"

    def __init__(self) -> None:
        super().__init__()
        self._frames_seen = 0

    def update(self, features, phase, ts_ms, dt_ms):
        self._frames_seen += 1

    def evaluate_rep(self, summary: RepSummary) -> Optional[ErrorReport]:
        if summary.phase_durations_ms.get("pull", 0) < 500:
            return ErrorReport(
                error_id=self.error_id, area=3, confidence=0.5,
                evidence="pull too short",
            )
        return None

    def reset(self) -> None:
        self._frames_seen = 0


def _feat(T=20.0) -> DeadliftFeatures:
    return DeadliftFeatures(
        side="left", T=T, K=160.0, back_angle=90 - T,
        hip_y=0.5, shoulder_y=0.25, avg_visibility=0.9, valid=True,
    )


def test_missing_error_id_raises():
    """error_id set edilmemis subclass instantiate edilirken patlar."""
    class Bad(ErrorDetector):
        # error_id unutulmus
        def evaluate_rep(self, s): return None
        def reset(self): pass

    with pytest.raises(TypeError, match="error_id"):
        Bad()


def test_default_update_is_noop():
    """Default update override edilmeden subclass calismali."""
    class RepOnly(ErrorDetector):
        error_id = "rep_only"
        def evaluate_rep(self, s): return None
        def reset(self): pass

    d = RepOnly()
    # Hatasiz cagri
    d.update(_feat(), Phase.PULL, ts_ms=100, dt_ms=33)


def test_dummy_detector_full_lifecycle():
    d = _DummyRepDetector()
    for ts in range(0, 200, 33):
        d.update(_feat(), Phase.PULL, ts_ms=ts, dt_ms=33)
    assert d._frames_seen > 0

    # PULL kisa -> hata
    summary_short = RepSummary(rep_id=1, phase_durations_ms={"pull": 200})
    rep = d.evaluate_rep(summary_short)
    assert rep is not None
    assert rep.error_id == "dummy_rep"
    assert rep.area == 3

    # PULL yeterince uzun -> hata yok
    summary_ok = RepSummary(rep_id=1, phase_durations_ms={"pull": 1000})
    assert d.evaluate_rep(summary_ok) is None

    d.reset()
    assert d._frames_seen == 0


# ---- MultiFrameConfirmation ----

def test_confirmation_requires_sustained_duration():
    """150ms boyunca surekli active olmali."""
    c = MultiFrameConfirmation(required_ms=150)
    assert c.update(True, ts_ms=0) is False        # baslangic
    assert c.update(True, ts_ms=100) is False       # 100ms sustained
    assert c.update(True, ts_ms=149) is False       # 149ms hala yetmez
    assert c.update(True, ts_ms=150) is True        # 150ms tam onay


def test_confirmation_resets_on_signal_drop():
    c = MultiFrameConfirmation(required_ms=150)
    c.update(True, ts_ms=0)
    c.update(True, ts_ms=100)
    assert c.update(False, ts_ms=120) is False      # sinyal kayboldu
    # Yeni sinyal sifirdan baslamali
    assert c.update(True, ts_ms=200) is False
    assert c.update(True, ts_ms=350) is True        # 200'den 350 = 150ms


def test_confirmation_tracks_max_sustained():
    c = MultiFrameConfirmation(required_ms=150)
    c.update(True, ts_ms=0)
    c.update(True, ts_ms=100)
    c.update(True, ts_ms=200)        # 200ms sustained
    c.update(False, ts_ms=210)       # drop
    c.update(True, ts_ms=300)
    c.update(True, ts_ms=350)        # 50ms sustained
    assert c.max_sustained_ms == 200


def test_confirmation_reset_clears_state():
    c = MultiFrameConfirmation(required_ms=150)
    c.update(True, ts_ms=0)
    c.update(True, ts_ms=200)
    assert c.max_sustained_ms > 0
    c.reset()
    assert c.max_sustained_ms == 0
    # Reset sonrasi yeni sinyal sifirdan baslamali
    assert c.update(True, ts_ms=1000) is False
    assert c.update(True, ts_ms=1150) is True
