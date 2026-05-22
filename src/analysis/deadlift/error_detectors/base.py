"""
Hata tespit modulleri icin ortak iskelet.

Tasarim:
  - Her hata bir ErrorDetector subclass'i (kendi dosyasinda).
  - Frame basina update() ile state biriktirir.
  - Rep kapandiginda evaluate_rep() ile karar verir (Optional[ErrorReport]).
  - reset() yeni rep'e gecmeden once cagrilir.

Squat anti-pattern'inden kacis (CLAUDE.md):
  - Karar mantigi tek bir abstract class altinda - dagilmiyor.
  - Multi-frame onay (120-200ms) MultiFrameConfirmation helper'inda;
    her detektor yeniden yazmiyor.
  - Cikti binary degil: ErrorReport area + confidence + evidence tasiyor.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

from src.analysis.deadlift.deadlift_features import DeadliftFeatures
from src.analysis.deadlift.phase_detector import Phase


# ---- ortak dataclass'lar ----

@dataclass(frozen=True)
class ErrorReport:
    """Tek bir hatanin rep-level karari."""
    error_id: str           # "incomplete_lockout", "uncontrolled_descent"
    area: int               # 1-5  (scoring.area_from_z)
    confidence: float       # 0.0-1.0  (scoring.confidence_from_z)
    evidence: str           # insan okunur kanit: "T_min=8.3 deg > 5 deg threshold"


@dataclass(frozen=True)
class RepSummary:
    """Rep tamamlandiginda detektore verilen baglam paketi.

    Detektorler kendi frame-level state'lerini zaten tutuyor; bu paket
    RepCounter'in saydigi (ve dedektorlerin tek basina hesaplayamayacagi)
    bilgilerin tasiyicisi.
    """
    rep_id: int
    phase_durations_ms: Dict[str, int]    # {"setup": ms, "pull": ms, ...}


# ---- detector ABC ----

class ErrorDetector(ABC):
    """
    Her hata bir subclass. Class-level `error_id` zorunlu.

    Yasam dongusu (analyzer tarafindan yonetilir):
      for frame:
          detector.update(features, phase, ts_ms, dt_ms)
      on rep close:
          report = detector.evaluate_rep(summary)
          detector.reset()
    """
    error_id: str = ""    # subclass override etmek zorunda

    def __init__(self) -> None:
        if not self.error_id:
            raise TypeError(
                f"{type(self).__name__} must set class attribute error_id"
            )

    def update(
        self,
        features: DeadliftFeatures,
        phase: Phase,
        ts_ms: int,
        dt_ms: int,
    ) -> None:
        """Frame basina cagrilir. Default: no-op (rep-level detektorler icin)."""
        # Frame-level state biriktirmek isteyen subclass override eder.
        return None

    @abstractmethod
    def evaluate_rep(self, summary: RepSummary) -> Optional[ErrorReport]:
        """
        Rep kapandiginda cagrilir.
        None doner -> bu rep'te bu hata YOK (raporlama).
        ErrorReport doner -> hata var, area+confidence+evidence dolu.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Yeni rep baslangici. Detektor frame-level state'ini temizler."""
        ...


# ---- multi-frame onay yardimcisi ----

@dataclass
class MultiFrameConfirmation:
    """
    Bir sinyal 'required_ms' boyunca kesintisiz aktifse onaylanir.
    Squat'tan oduc alinan _hold pattern'inin temizi.

    Kullanim:
        conf = MultiFrameConfirmation(required_ms=150)
        for frame:
            confirmed = conf.update(signal_active=bool(...), ts_ms=ts)
        # confirmed True olursa hata 150ms sustained demektir.

    `max_sustained_ms` rep boyunca gozlenen en uzun kesintisiz aktif suresi.
    Rep evaluation'da bu kullanilarak confidence olceklenir.
    """
    required_ms: int = 150
    _signal_start_ts: Optional[int] = None
    _max_sustained_ms: int = 0

    def update(self, signal_active: bool, ts_ms: int) -> bool:
        """Sinyalin durumunu ilet, su an onayli mi (>= required_ms) doner."""
        if signal_active:
            if self._signal_start_ts is None:
                self._signal_start_ts = ts_ms
            sustained = ts_ms - self._signal_start_ts
            if sustained > self._max_sustained_ms:
                self._max_sustained_ms = sustained
            return sustained >= self.required_ms
        else:
            self._signal_start_ts = None
            return False

    @property
    def max_sustained_ms(self) -> int:
        return self._max_sustained_ms

    def reset(self) -> None:
        self._signal_start_ts = None
        self._max_sustained_ms = 0
