"""
incomplete_lockout: LOCKOUT fazinda T'nin minimumu > 5 derece ise hata.

Sinyal: LOCKOUT boyunca min(T) izlenir.
Karar: pseudo-Z = max(0, (T_min - 5) / 1.5) -> area + confidence.

Multi-frame onay gereksiz: LOCKOUT'a giris zaten 300ms hold ile onayli
(PhaseDetector.LOCKOUT_HOLD_MS).
"""
from __future__ import annotations

from typing import Optional

from src.analysis.deadlift.deadlift_features import DeadliftFeatures
from src.analysis.deadlift.error_detectors.base import (
    ErrorDetector,
    ErrorReport,
    RepSummary,
)
from src.analysis.deadlift.phase_detector import Phase
from src.analysis.deadlift.scoring import area_from_z, confidence_from_z


T_LOCKOUT_THRESHOLD: float = 5.0
T_TOLERANCE: float = 1.5     # pseudo-Z paydasi (1 standart sapma kabulu)


class IncompleteLockoutDetector(ErrorDetector):
    error_id = "incomplete_lockout"

    def __init__(self) -> None:
        super().__init__()
        self._t_min_lockout: Optional[float] = None

    def update(
        self,
        features: DeadliftFeatures,
        phase: Phase,
        ts_ms: int,
        dt_ms: int,
    ) -> None:
        if phase != Phase.LOCKOUT:
            return
        if not features.valid:
            return
        if self._t_min_lockout is None or features.T < self._t_min_lockout:
            self._t_min_lockout = features.T

    def evaluate_rep(self, summary: RepSummary) -> Optional[ErrorReport]:
        if self._t_min_lockout is None:
            # LOCKOUT hic gozlenmedi -> incomplete rep, RepCounter zaten halletti.
            return None

        t_min = self._t_min_lockout
        pseudo_z = max(0.0, (t_min - T_LOCKOUT_THRESHOLD) / T_TOLERANCE)
        area = area_from_z(pseudo_z)
        confidence = confidence_from_z(pseudo_z)

        if t_min <= T_LOCKOUT_THRESHOLD:
            evidence = f"T_min={t_min:.1f} deg (within {T_LOCKOUT_THRESHOLD:.1f} threshold)"
        else:
            evidence = (
                f"T_min={t_min:.1f} deg > {T_LOCKOUT_THRESHOLD:.1f} threshold "
                f"(pseudo_z={pseudo_z:.2f})"
            )

        return ErrorReport(
            error_id=self.error_id,
            area=area,
            confidence=confidence,
            evidence=evidence,
        )

    def reset(self) -> None:
        self._t_min_lockout = None
