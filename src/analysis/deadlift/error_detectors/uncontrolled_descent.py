"""
uncontrolled_descent: kontrolsuz iniş.

Iki bagimsiz sinyal, OR semantigi:

  A) Sure orani  (rep-level)
     descent_duration_ms / pull_duration_ms < 0.5
     -> rep kapaninda hesaplanir, multi-frame onay gerekmez (zaten aggregate).

  B) Hip hizi surekliligi  (frame-level)
     DESCENT sirasinda |v_hip| > esik degeri en az 150ms surdu mu?
     MultiFrameConfirmation kullanir.

Notlar:
  - Spec "ivme" diyor (CLAUDE.md:143). Ikinci turev gurultuye duyarli oldugundan
    V0'da "esikli hiz + sustained" deyimi kullanildi. Sprint 5 validation
    sonucuna gore ivme tabanli bir varyant denenebilir.
  - HIP_VELOCITY_THRESHOLD heuristic; Sprint 5'te 15-20 video uzerinde
    kalibre edilecek. Yanlissa false_positive_rate > 0.20 -> V1 trigger.
  - Iki sinyalin pseudo-Z'lerinin max'i alinir (OR semantigi).
"""
from __future__ import annotations

from typing import Optional

from src.analysis.deadlift.deadlift_features import DeadliftFeatures
from src.analysis.deadlift.error_detectors.base import (
    ErrorDetector,
    ErrorReport,
    MultiFrameConfirmation,
    RepSummary,
)
from src.analysis.deadlift.phase_detector import Phase
from src.analysis.deadlift.scoring import area_from_z, confidence_from_z


# --- esikler (Sprint 5'te kalibre edilecek heuristic'ler) ---
RATIO_THRESHOLD: float = 0.5
RATIO_TOLERANCE: float = 0.10        # pseudo-Z paydasi

HIP_VELOCITY_THRESHOLD_PER_MS: float = 4e-4
SUSTAINED_DROP_MS: int = 150


class UncontrolledDescentDetector(ErrorDetector):
    error_id = "uncontrolled_descent"

    def __init__(self) -> None:
        super().__init__()
        self._confirmation = MultiFrameConfirmation(required_ms=SUSTAINED_DROP_MS)
        self._prev_hip_y: Optional[float] = None
        self._prev_ts: Optional[int] = None
        self._peak_velocity: float = 0.0     # mutlak deger, evidence icin

    def update(
        self,
        features: DeadliftFeatures,
        phase: Phase,
        ts_ms: int,
        dt_ms: int,
    ) -> None:
        # Sadece DESCENT fazinda v_hip izle
        if phase != Phase.DESCENT or not features.valid:
            # Prev'i tut ki DESCENT'e geri donulurse turev hesabi devam etsin
            self._prev_hip_y = features.hip_y if features.valid else self._prev_hip_y
            self._prev_ts = ts_ms
            return

        if self._prev_hip_y is None or self._prev_ts is None:
            self._prev_hip_y = features.hip_y
            self._prev_ts = ts_ms
            return

        dt = max(1, ts_ms - self._prev_ts)
        v_hip = (features.hip_y - self._prev_hip_y) / dt    # 1/ms; pozitif = asagi
        abs_v = abs(v_hip)
        if abs_v > self._peak_velocity:
            self._peak_velocity = abs_v

        # Eşik üstünde mi? Sustained kontrolu MultiFrameConfirmation'da.
        signal_active = abs_v > HIP_VELOCITY_THRESHOLD_PER_MS
        self._confirmation.update(signal_active, ts_ms)

        self._prev_hip_y = features.hip_y
        self._prev_ts = ts_ms

    def evaluate_rep(self, summary: RepSummary) -> Optional[ErrorReport]:
        descent_ms = summary.phase_durations_ms.get("descent", 0)
        if descent_ms <= 0:
            # DESCENT hic gozlenmedi -> RepCounter zaten incomplete dedi.
            return None

        # --- Sinyal A: sure orani ---
        pull_ms = summary.phase_durations_ms.get("pull", 0)
        ratio: Optional[float] = None
        z_a = 0.0
        if pull_ms > 0:
            ratio = descent_ms / pull_ms
            z_a = max(0.0, (RATIO_THRESHOLD - ratio) / RATIO_TOLERANCE)

        # --- Sinyal B: sustained hip drop ---
        sustained_ms = self._confirmation.max_sustained_ms
        z_b = 0.0
        if sustained_ms >= SUSTAINED_DROP_MS:
            z_b = sustained_ms / SUSTAINED_DROP_MS   # 150ms = 1.0, 600ms = 4.0

        # --- Birlestir (OR -> max) ---
        z = max(z_a, z_b)
        area = area_from_z(z)
        confidence = confidence_from_z(z)

        evidence = self._build_evidence(
            ratio=ratio, descent_ms=descent_ms, pull_ms=pull_ms,
            sustained_ms=sustained_ms,
            peak_v=self._peak_velocity,
            z_a=z_a, z_b=z_b,
        )

        return ErrorReport(
            error_id=self.error_id,
            area=area, confidence=confidence,
            evidence=evidence,
        )

    def reset(self) -> None:
        self._confirmation.reset()
        self._prev_hip_y = None
        self._prev_ts = None
        self._peak_velocity = 0.0

    # ---- internal ----

    @staticmethod
    def _build_evidence(
        ratio: Optional[float], descent_ms: int, pull_ms: int,
        sustained_ms: int, peak_v: float, z_a: float, z_b: float,
    ) -> str:
        parts = []
        if ratio is not None:
            parts.append(
                f"descent/pull={ratio:.2f} ({descent_ms}/{pull_ms}ms, z={z_a:.2f})"
            )
        else:
            parts.append(f"descent={descent_ms}ms, pull=0ms (ratio n/a)")
        parts.append(
            f"sustained_drop={sustained_ms}ms "
            f"(peak_v={peak_v:.2e} 1/ms, z={z_b:.2f})"
        )
        return " | ".join(parts)
