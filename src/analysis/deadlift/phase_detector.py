"""
Katman 4 - Faz tespiti (state machine).
Faz sirasi: SETUP -> PULL -> LOCKOUT -> DESCENT -> SETUP

Tum gecisler timestamp_ms uzerinden hesaplanir (frame sayisi DEGIL).
Squat anti-pattern'inden kacis: her gecis multi-frame onayli.
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from src.analysis.deadlift.calibration import StandingBaseline
from src.analysis.deadlift.deadlift_features import DeadliftFeatures

logger = logging.getLogger(__name__)


class Phase(str, Enum):
    SETUP = "setup"
    PULL = "pull"
    LOCKOUT = "lockout"
    DESCENT = "descent"


@dataclass(frozen=True)
class PhaseEvent:
    phase: Phase
    phase_changed: bool
    ts_ms: int
    from_phase: Optional[Phase] = None   # sadece phase_changed=True iken doldurulur
    hip_velocity: float = 0.0            # 1/ms cinsinden
    T: float = 0.0


# --- Esikler (tek noktada, ileride reference_distributions.json'a tasinabilir) ---
T_SETUP_MIN = 30.0            # SETUP'ta T > 30°
T_LOCKOUT_MAX = 8.0           # LOCKOUT'ta T < 8° (landmark titremesi icin pay)
HIP_NEAR_STANDING_TOL = 0.03  # standing_hip_y +/- tolerans
HIP_SETUP_DELTA = 0.10        # hip standing'den en az bu kadar asagida -> SETUP
VELOCITY_EPS = 5e-5           # 1/ms cinsinden
LOCKOUT_HOLD_MS = 300         # CLAUDE.md - 300ms sabit
MIN_PHASE_HOLD_MS = 80        # diger gecisler icin minimum onay suresi


@dataclass
class PhaseDetector:
    baseline: StandingBaseline

    phase: Phase = Phase.SETUP
    phase_start_ts: Optional[int] = None
    history: List[Phase] = field(default_factory=list)

    _prev_hip_y: Optional[float] = None
    _prev_ts: Optional[int] = None
    _candidate_phase: Optional[Phase] = None
    _candidate_start_ts: Optional[int] = None

    def update(self, features: DeadliftFeatures, ts_ms: int) -> PhaseEvent:
        if not features.valid:
            return PhaseEvent(phase=self.phase, phase_changed=False, ts_ms=ts_ms)

        v_hip = self._velocity(features.hip_y, ts_ms)
        self._prev_hip_y = features.hip_y
        self._prev_ts = ts_ms

        if self.phase_start_ts is None:
            self.phase_start_ts = ts_ms

        candidate = self._classify(features, v_hip)
        changed, from_phase = self._maybe_transition(candidate, ts_ms)

        return PhaseEvent(
            phase=self.phase,
            phase_changed=changed,
            ts_ms=ts_ms,
            from_phase=from_phase,
            hip_velocity=v_hip,
            T=features.T,
        )

    def _velocity(self, hip_y: float, ts_ms: int) -> float:
        if self._prev_hip_y is None or self._prev_ts is None:
            return 0.0
        dt = ts_ms - self._prev_ts
        if dt <= 0:
            return 0.0
        return (hip_y - self._prev_hip_y) / dt

    def _classify(self, f: DeadliftFeatures, v_hip: float) -> Phase:
        """Anlik faz adayi - henuz commit edilmemis."""
        hip_y = f.hip_y
        T = f.T
        standing_y = self.baseline.standing_hip_y

        # LOCKOUT: govde dik (T cok kucuk) + hareketsiz.
        # NOT: hip standing'e yakin mi kontrolu KULLANILMIYOR cunku kalibrasyon
        # genelde setup pozisyonunda yapiliyor; T+velocity zaten yeterli ayrim.
        if T < T_LOCKOUT_MAX and abs(v_hip) < VELOCITY_EPS:
            return Phase.LOCKOUT

        # SETUP: hip belirgin asagida + T buyuk + hareketsiz
        if (
            hip_y > standing_y + HIP_SETUP_DELTA
            and T > T_SETUP_MIN
            and abs(v_hip) < VELOCITY_EPS
        ):
            return Phase.SETUP

        # PULL: hip yukseliyor (y azaliyor - MediaPipe'da y asagi pozitif)
        if v_hip < -VELOCITY_EPS:
            return Phase.PULL

        # DESCENT: hip iniyor (y artiyor)
        if v_hip > VELOCITY_EPS:
            return Phase.DESCENT

        # Hicbiri tetiklenmediyse mevcut fazda kal
        return self.phase

    def _maybe_transition(
        self, candidate: Phase, ts_ms: int
    ) -> Tuple[bool, Optional[Phase]]:
        """Adayi multi-frame/ms onayindan gecirip commit eder."""
        if candidate == self.phase:
            self._candidate_phase = None
            self._candidate_start_ts = None
            return False, None

        if candidate != self._candidate_phase:
            self._candidate_phase = candidate
            self._candidate_start_ts = ts_ms
            return False, None

        hold_required = (
            LOCKOUT_HOLD_MS if candidate == Phase.LOCKOUT else MIN_PHASE_HOLD_MS
        )
        held = ts_ms - (self._candidate_start_ts or ts_ms)
        if held < hold_required:
            return False, None

        if not self._is_legal_transition(self.phase, candidate):
            logger.debug(
                "Illegal gecis: %s -> %s (ignore)",
                self.phase.value, candidate.value,
            )
            self._candidate_phase = None
            self._candidate_start_ts = None
            return False, None

        from_phase = self.phase
        logger.info(
            "Faz gecisi: %s -> %s @ %d ms (held %d ms)",
            from_phase.value, candidate.value, ts_ms, held,
        )
        self.history.append(from_phase)
        self.phase = candidate
        self.phase_start_ts = ts_ms
        self._candidate_phase = None
        self._candidate_start_ts = None
        return True, from_phase

    @staticmethod
    def _is_legal_transition(current: Phase, nxt: Phase) -> bool:
        """SETUP -> PULL -> LOCKOUT -> DESCENT -> SETUP dongusu.
        DESCENT -> PULL ardisik rep'lere izin verir."""
        legal = {
            Phase.SETUP: {Phase.PULL},
            Phase.PULL: {Phase.LOCKOUT, Phase.DESCENT},
            Phase.LOCKOUT: {Phase.DESCENT},
            Phase.DESCENT: {Phase.SETUP, Phase.PULL},
        }
        return nxt in legal[current]
    
    def get_diagnostics(self, ts_ms: int) -> dict:
        held = 0
        if self._candidate_start_ts is not None:
            held = ts_ms - self._candidate_start_ts
        return {
            "candidate_phase": self._candidate_phase.value if self._candidate_phase else None,
            "candidate_held_ms": held,
        }