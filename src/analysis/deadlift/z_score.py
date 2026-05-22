"""
Katman 6 - Z-skor hesaplama.

Bir CheckpointSample + onun reference dagilimi -> her olcum kanali icin z.
Saf fonksiyon; state yok, karar yok. Hata mantigi Sprint 4'un isi.

Referans veriler reference_distributions.json'dan yuklenir (schema 0.2.x).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from src.analysis.deadlift.checkpoints import CheckpointSample

logger = logging.getLogger(__name__)

REFERENCE_PATH = Path(__file__).parent / "reference_distributions.json"
SUPPORTED_SCHEMA_PREFIX = "0.2."

@dataclass(frozen=True)
class CheckpointReference:
    """Bir checkpoint icin referans dagilim (literatur tabanli V0)."""
    id: str
    T: float
    K_mean: float
    K_std: float
    hip_shoulder_velocity_ratio_mean: float
    hip_shoulder_velocity_ratio_std: float

@dataclass(frozen=True)
class ZScores:
    """Bir CheckpointSample icin hesaplanmis z degerleri."""
    checkpoint_id: str
    K_z: float
    velocity_ratio_z: Optional[float]   # None = sample.velocity_ratio_valid False idi

# ---- loader ----

def load_reference(path: Path = REFERENCE_PATH) -> Dict[str, CheckpointReference]:
    """
    JSON'u id-keyed dict olarak yukle.

    Hatalar:
      - schema_version 0.2.x degilse ValueError
      - 14 ckpt beklenir (7 PULL + 7 DESCENT); eksikse ValueError
      - K_std = 0 -> ValueError (bolme hatasini erken yakala)
    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    version = raw.get("_meta", {}).get("schema_version", "")
    if not version.startswith(SUPPORTED_SCHEMA_PREFIX):
        raise ValueError(
            f"Unsupported reference schema {version!r}; "
            f"expected {SUPPORTED_SCHEMA_PREFIX}x"
        )

    out: Dict[str, CheckpointReference] = {}
    for phase_block in raw["phases"].values():
        for ckpt in phase_block["checkpoints"]:
            ref = CheckpointReference(
                id=ckpt["id"],
                T=float(ckpt["T"]),
                K_mean=float(ckpt["K_mean"]),
                K_std=float(ckpt["K_std"]),
                hip_shoulder_velocity_ratio_mean=float(
                    ckpt["hip_shoulder_velocity_ratio_mean"]
                ),
                hip_shoulder_velocity_ratio_std=float(
                    ckpt["hip_shoulder_velocity_ratio_std"]
                ),
            )
            if ref.K_std == 0.0:
                raise ValueError(f"K_std=0 for {ref.id}; cannot z-score")
            if ref.hip_shoulder_velocity_ratio_std == 0.0:
                raise ValueError(
                    f"velocity_ratio_std=0 for {ref.id}; cannot z-score"
                )
            out[ref.id] = ref

    if len(out) != 14:
        raise ValueError(
            f"Expected 14 checkpoints in reference, got {len(out)}: "
            f"{sorted(out.keys())}"
        )
    return out

# ---- hesaplama ----

def compute_z(sample: CheckpointSample, ref: CheckpointReference) -> ZScores:
    """
    Saf fonksiyon: sample + ref -> ZScores.

    Caller sample.id ile reference dict'inden ref'i secip cagirir.
    sample.id == ref.id varsayilir (mismatch ValueError).
    """
    if sample.id != ref.id:
        raise ValueError(
            f"Sample id {sample.id!r} mismatches reference id {ref.id!r}"
        )

    K_z = (sample.K - ref.K_mean) / ref.K_std

    if sample.velocity_ratio_valid:
        velocity_z = (
            sample.hip_shoulder_velocity_ratio
            - ref.hip_shoulder_velocity_ratio_mean
        ) / ref.hip_shoulder_velocity_ratio_std
    else:
        velocity_z = None

    return ZScores(
        checkpoint_id=sample.id, K_z=K_z, velocity_ratio_z=velocity_z,
    )
