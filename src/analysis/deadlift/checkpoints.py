"""
Katman 5 - T-acisi checkpoint ornekleme.

Her frame'i degerlendirmek yerine, govdenin onceden tanimli T acilarinda
(35, 30, 25, 20, 15, 10, 5) oldugu anlardaki frame'i yakala. PULL ve DESCENT
fazlari icin ayri ayri. Yakalanan her ckpt icin K + hip_shoulder_velocity_ratio
olc, downstream Z-score icin disari ver.

Squat'tan farki: frame-anlik karar yok, icerik-tabanli ornekleme.

Edge case'ler (CLAUDE.md Katman 5):
  - Hizli gecis: |T_prev - T_curr| > 3 deg ise aradaki target'lari lineer
    interpole et (interpolated=True isaretlenir).
  - Coklu tetikleme: ayni ckpt birden fazla frame'de tolerans bandinda olursa,
    T'nin target'a en yakin oldugu frame secilir. Banttan cikinca ckpt
    finalize edilir (emit).
  - Velocity ratio: omuz dikey hizi epsilon altindaysa ratio tanimsiz
    (velocity_ratio_valid=False).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.analysis.deadlift.deadlift_features import DeadliftFeatures
from src.analysis.deadlift.phase_detector import Phase

logger = logging.getLogger(__name__)


CHECKPOINT_T_DEGREES: Tuple[int, ...] = (35, 30, 25, 20, 15, 10, 5)
TOLERANCE_DEG: float = 1.5
FAST_TRANSITION_DEG: float = 3.0
SHOULDER_VELOCITY_EPS: float = 1e-6  # 1/ms; bunun altinda ratio tanimsiz


@dataclass(frozen=True)
class CheckpointSample:
    id: str                                 # "PULL_35", "DESCENT_10" ...
    phase: Phase                            # PULL veya DESCENT
    T_target: float                         # 35.0, 30.0, ...
    T_observed: float                       # gercek olculen / interpolasyondan
    K: float
    hip_shoulder_velocity_ratio: float
    velocity_ratio_valid: bool              # False ise downstream ignore etmeli
    timestamp_ms: int
    interpolated: bool                      # iki frame arasi sentez mi


@dataclass
class _PendingCkpt:
    """Tolerans bandindayken biriken en iyi (target'a en yakin) sample."""
    best_sample: CheckpointSample
    best_abs_delta: float


@dataclass
class CheckpointCollector:
    """
    Stateful. Faz disinda (SETUP/LOCKOUT) update sessizce frame'i izler ama
    ckpt uretmez. Rep-level reset() disaridan cagrilir (analyzer rep
    tamamlandiginda).

    update() yakalanan ckpt'leri liste olarak dondurur (genellikle 0 veya 1;
    hizli gecis durumunda 2+).
    """
    _prev: Optional[DeadliftFeatures] = None
    _prev_ts: Optional[int] = None
    _locked: Dict[str, CheckpointSample] = field(default_factory=dict)
    _pending: Dict[str, _PendingCkpt] = field(default_factory=dict)
    _current_collection_phase: Optional[Phase] = None

    # ---- public API ----

    def reset(self) -> None:
        """Yeni rep baslangicinda cagir. Tum durum temizlenir."""
        self._prev = None
        self._prev_ts = None
        self._locked.clear()
        self._pending.clear()
        self._current_collection_phase = None

    def collected(self) -> List[CheckpointSample]:
        """Bu rep'te su ana kadar finalize edilmis ckpt'ler (deterministik sirayla)."""
        return [self._locked[i] for i in sorted(self._locked.keys())]

    def update(
        self,
        phase: Phase,
        features: DeadliftFeatures,
        ts_ms: int,
    ) -> List[CheckpointSample]:
        """Frame basina cagrilir. Yeni finalize edilen ckpt(ler)i dondurur."""
        if not features.valid:
            return []

        emitted: List[CheckpointSample] = []

        if phase not in (Phase.PULL, Phase.DESCENT):
            # PULL veya DESCENT'e tekrar girilirken pending'ler kapatilmali.
            emitted.extend(self._flush_pending())
            self._prev = features
            self._prev_ts = ts_ms
            self._current_collection_phase = None
            return emitted

        # Faz degisikligi (PULL -> DESCENT veya tersi): pending'leri kapat,
        # icindeki ckpt'ler eski faza aittir, emit'le.
        if (
            self._current_collection_phase is not None
            and self._current_collection_phase != phase
        ):
            emitted.extend(self._flush_pending())

        self._current_collection_phase = phase

        # 1) Hizli gecis interpolasyonu
        if self._prev is not None and self._prev_ts is not None:
            dT = abs(features.T - self._prev.T)
            if dT > FAST_TRANSITION_DEG:
                emitted.extend(
                    self._interpolate_missed(phase, features, ts_ms)
                )

        # 2) Direkt match / coklu tetikleme
        emitted.extend(self._check_band(phase, features, ts_ms))

        self._prev = features
        self._prev_ts = ts_ms
        return emitted

    # ---- internal ----

    def _ckpt_id(self, phase: Phase, target_T: int) -> str:
        return f"{phase.value.upper()}_{target_T}"

    def _check_band(
        self,
        phase: Phase,
        features: DeadliftFeatures,
        ts_ms: int,
    ) -> List[CheckpointSample]:
        emitted: List[CheckpointSample] = []
        T = features.T

        for target in CHECKPOINT_T_DEGREES:
            ckpt_id = self._ckpt_id(phase, target)
            if ckpt_id in self._locked:
                continue

            delta = T - target
            abs_delta = abs(delta)
            in_band = abs_delta < TOLERANCE_DEG

            if in_band:
                sample = self._build_sample(
                    phase=phase,
                    target_T=float(target),
                    observed_T=T,
                    K=features.K,
                    hip_y=features.hip_y,
                    shoulder_y=features.shoulder_y,
                    ts_ms=ts_ms,
                    interpolated=False,
                )
                pending = self._pending.get(ckpt_id)
                if pending is None or abs_delta < pending.best_abs_delta:
                    self._pending[ckpt_id] = _PendingCkpt(
                        best_sample=sample, best_abs_delta=abs_delta,
                    )
            else:
                # Bantta degil; pending varsa finalize et.
                pending = self._pending.pop(ckpt_id, None)
                if pending is not None:
                    self._locked[ckpt_id] = pending.best_sample
                    emitted.append(pending.best_sample)
                    logger.debug(
                        "Ckpt finalize: %s @ T=%.2f (target=%d)",
                        ckpt_id, pending.best_sample.T_observed, target,
                    )
        return emitted

    def _interpolate_missed(
        self,
        phase: Phase,
        curr: DeadliftFeatures,
        ts_ms: int,
    ) -> List[CheckpointSample]:
        """
        prev_T ile curr_T arasinda bant atlanmis target'lari sentezle.
        Faz PULL ise T dususte, DESCENT ise T yukseliste. Iki yon de desteklenir
        - sirf [min, max] araliginda olup henuz locked olmayan target'lar gecer.
        """
        assert self._prev is not None and self._prev_ts is not None
        prev_T = self._prev.T
        curr_T = curr.T
        if prev_T == curr_T:
            return []

        emitted: List[CheckpointSample] = []
        lo, hi = sorted((prev_T, curr_T))
        for target in CHECKPOINT_T_DEGREES:
            ckpt_id = self._ckpt_id(phase, target)
            if ckpt_id in self._locked:
                continue
            # Sadece prev/curr arasinda kalan target'lar; pending varsa o da
            # interpolasyona yenik dusebilir mi? Hayir, pending = bantta zaten
            # gozlenmis demek; interpolasyon yapma, pending dogal yoldan
            # finalize olur.
            if ckpt_id in self._pending:
                continue
            if not (lo < target < hi):
                continue

            f = (target - prev_T) / (curr_T - prev_T)  # [0, 1]
            K_interp = self._prev.K + f * (curr.K - self._prev.K)
            hip_y_interp = self._prev.hip_y + f * (curr.hip_y - self._prev.hip_y)
            shoulder_y_interp = (
                self._prev.shoulder_y
                + f * (curr.shoulder_y - self._prev.shoulder_y)
            )
            ts_interp = int(round(self._prev_ts + f * (ts_ms - self._prev_ts)))

            # Velocity ratio: interpolasyon noktasinda da prev->curr toplam
            # hizini kullan (kisa pencerede sabit varsayim).
            sample = self._build_sample_from_velocities(
                phase=phase,
                target_T=float(target),
                observed_T=float(target),  # tam target degerinde sentez
                K=K_interp,
                v_hip=(curr.hip_y - self._prev.hip_y) / max(1, ts_ms - self._prev_ts),
                v_shoulder=(curr.shoulder_y - self._prev.shoulder_y)
                           / max(1, ts_ms - self._prev_ts),
                ts_ms=ts_interp,
                interpolated=True,
            )
            self._locked[ckpt_id] = sample
            emitted.append(sample)
            logger.debug(
                "Ckpt interpolasyon: %s @ T=%d (prev=%.2f, curr=%.2f, f=%.2f)",
                ckpt_id, target, prev_T, curr_T, f,
            )
        return emitted

    def _flush_pending(self) -> List[CheckpointSample]:
        """Tolerans bandinda kalmis pending'leri finalize et."""
        emitted: List[CheckpointSample] = []
        for ckpt_id, pending in self._pending.items():
            if ckpt_id in self._locked:
                continue
            self._locked[ckpt_id] = pending.best_sample
            emitted.append(pending.best_sample)
        self._pending.clear()
        return emitted

    def _build_sample(
        self,
        phase: Phase,
        target_T: float,
        observed_T: float,
        K: float,
        hip_y: float,
        shoulder_y: float,
        ts_ms: int,
        interpolated: bool,
    ) -> CheckpointSample:
        if self._prev is None or self._prev_ts is None:
            return CheckpointSample(
                id=self._ckpt_id(phase, int(target_T)),
                phase=phase, T_target=target_T, T_observed=observed_T, K=K,
                hip_shoulder_velocity_ratio=0.0,
                velocity_ratio_valid=False,
                timestamp_ms=ts_ms, interpolated=interpolated,
            )
        dt = max(1, ts_ms - self._prev_ts)
        v_hip = (hip_y - self._prev.hip_y) / dt
        v_shoulder = (shoulder_y - self._prev.shoulder_y) / dt
        return self._build_sample_from_velocities(
            phase=phase, target_T=target_T, observed_T=observed_T, K=K,
            v_hip=v_hip, v_shoulder=v_shoulder, ts_ms=ts_ms,
            interpolated=interpolated,
        )

    def _build_sample_from_velocities(
        self,
        phase: Phase,
        target_T: float,
        observed_T: float,
        K: float,
        v_hip: float,
        v_shoulder: float,
        ts_ms: int,
        interpolated: bool,
    ) -> CheckpointSample:
        valid = abs(v_shoulder) >= SHOULDER_VELOCITY_EPS
        ratio = v_hip / v_shoulder if valid else 0.0
        return CheckpointSample(
            id=self._ckpt_id(phase, int(target_T)),
            phase=phase,
            T_target=target_T,
            T_observed=observed_T,
            K=K,
            hip_shoulder_velocity_ratio=ratio,
            velocity_ratio_valid=valid,
            timestamp_ms=ts_ms,
            interpolated=interpolated,
        )
    def start_new_rep(self) -> None:
        """
        Rep kapandiginda cagrilir. _locked ve _pending temizlenir; _prev ve
        _current_collection_phase korunur (sonraki frame'de velocity hesabi
        surekliliği icin).
        """
        self._locked.clear()
        self._pending.clear()
