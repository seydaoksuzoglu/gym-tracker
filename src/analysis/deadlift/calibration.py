"""
Katman 3 - Vucut orani normalizasyonu (standing baseline).
Ilk N valid frame'den torso_length, shoulder_width, femur_length hesaplar.
Squat'in aksine baseline TEK NOKTADA tutulur.
"""
import logging
import statistics
from dataclasses import dataclass
from typing import List, Optional

from src.analysis.deadlift.landmarks import LEFT, RIGHT, LandmarkFrame
from src.common.geometry import distance

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StandingBaseline:
    torso_length: float
    shoulder_width: float
    femur_length: float


class StandingCalibrator:
    """Ilk N valid frame'den standing baseline hesaplar."""

    def __init__(
        self,
        frames_required: int = 30,
        max_motion_std: float = 0.02,
    ):
        self.frames_required = frames_required
        self.max_motion_std = max_motion_std
        self._torso_samples: List[float] = []
        self._shoulder_samples: List[float] = []
        self._femur_samples: List[float] = []
        self._hip_y_samples: List[float] = []
        self._baseline: Optional[StandingBaseline] = None
        self._failed: bool = False

    def update(self, frame: LandmarkFrame) -> None:
        if self._baseline is not None or self._failed:
            return

        ls = frame.get(LEFT["shoulder"])
        rs = frame.get(RIGHT["shoulder"])
        lh = frame.get(LEFT["hip"])
        rh = frame.get(RIGHT["hip"])
        lk = frame.get(LEFT["knee"])
        rk = frame.get(RIGHT["knee"])

        # Omuzlar ve kalçalar zorunlu (midpoint ve shoulder_width için)
        if not all(p.valid for p in (ls, rs, lh, rh)):
            return

        # Femur: side view'da uzak diz tipik olarak gorunmuyor.
        # En az bir tarafin dizi valid'se onu kullan; her ikisi de varsa ortalama.
        femur_samples = []
        if lk.valid:
            femur_samples.append(distance(lh.as_xy(), lk.as_xy()))
        if rk.valid:
            femur_samples.append(distance(rh.as_xy(), rk.as_xy()))

        if not femur_samples:
            return  # Hicbir diz gorunmuyor, frame'i atla

        shoulder_mid = ((ls.x + rs.x) / 2.0, (ls.y + rs.y) / 2.0)
        hip_mid = ((lh.x + rh.x) / 2.0, (lh.y + rh.y) / 2.0)

        torso = distance(shoulder_mid, hip_mid)
        shoulder_w = distance(ls.as_xy(), rs.as_xy())
        femur = sum(femur_samples) / len(femur_samples)


        self._torso_samples.append(torso)
        self._shoulder_samples.append(shoulder_w)
        self._femur_samples.append(femur)
        self._hip_y_samples.append(hip_mid[1])

        if len(self._torso_samples) >= self.frames_required:
            self._finalize()

    def _finalize(self) -> None:
        # Hareket kontrolu
        if len(self._hip_y_samples) >= 2:
            hip_std = statistics.pstdev(self._hip_y_samples)
            if hip_std > self.max_motion_std:
                self._failed = True
                logger.warning(
                    "Kalibrasyon basarisiz: kullanici sabit degil "
                    "(hip.y std=%.4f > %.4f). 'Bar'in onunde 1 saniye sabit dur.'",
                    hip_std, self.max_motion_std,
                )
                return

        self._baseline = StandingBaseline(
            torso_length=statistics.fmean(self._torso_samples),
            shoulder_width=statistics.fmean(self._shoulder_samples),
            femur_length=statistics.fmean(self._femur_samples),
        )
        logger.info(
            "Kalibrasyon tamam: torso=%.4f shoulder=%.4f femur=%.4f",
            self._baseline.torso_length,
            self._baseline.shoulder_width,
            self._baseline.femur_length,
        )

    def is_ready(self) -> bool:
        return self._baseline is not None

    def has_failed(self) -> bool:
        return self._failed

    def get_baseline(self) -> StandingBaseline:
        if self._baseline is None:
            raise RuntimeError("Kalibrasyon henuz hazir degil. Once is_ready() kontrol et.")
        return self._baseline

    def reset(self) -> None:
        self._torso_samples.clear()
        self._shoulder_samples.clear()
        self._femur_samples.clear()
        self._hip_y_samples.clear()
        self._baseline = None
        self._failed = False
