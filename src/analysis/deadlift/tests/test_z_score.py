"""
load_reference + compute_z birim testleri.
"""
import json
from pathlib import Path

import pytest

from src.analysis.deadlift.checkpoints import CheckpointSample
from src.analysis.deadlift.phase_detector import Phase
from src.analysis.deadlift.z_score import (
    CheckpointReference,
    REFERENCE_PATH,
    ZScores,
    compute_z,
    load_reference,
)


# ---- load_reference ----

def test_load_reference_returns_14_checkpoints():
    refs = load_reference()
    assert len(refs) == 14
    expected_ids = {
        f"PULL_{t}" for t in (35, 30, 25, 20, 15, 10, 5)
    } | {
        f"DESCENT_{t}" for t in (35, 30, 25, 20, 15, 10, 5)
    }
    assert set(refs.keys()) == expected_ids


def test_load_reference_known_pull_25_values():
    """PULL_25 (knee pass) reference_distributions.json'a uygun olmali."""
    refs = load_reference()
    pull_25 = refs["PULL_25"]
    assert pull_25.T == 25.0
    assert pull_25.K_mean == 158.0
    assert pull_25.K_std == 7.0
    assert pull_25.hip_shoulder_velocity_ratio_mean == 0.98
    assert pull_25.hip_shoulder_velocity_ratio_std == 0.18


def test_load_reference_rejects_old_schema(tmp_path: Path):
    """schema_version 0.1.x -> ValueError."""
    fake = {
        "_meta": {"schema_version": "0.1.0"},
        "phases": {"PULL": {"checkpoints": []}, "DESCENT": {"checkpoints": []}},
    }
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(fake), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported reference schema"):
        load_reference(p)


def test_load_reference_rejects_zero_std(tmp_path: Path):
    """K_std=0 -> ValueError (bolme hatasini erken yakala)."""
    fake = {
        "_meta": {"schema_version": "0.2.0"},
        "phases": {
            "PULL": {
                "checkpoints": [{
                    "id": "PULL_35", "T": 35,
                    "K_mean": 140.0, "K_std": 0.0,
                    "hip_shoulder_velocity_ratio_mean": 1.0,
                    "hip_shoulder_velocity_ratio_std": 0.2,
                }],
            },
            "DESCENT": {"checkpoints": []},
        },
    }
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(fake), encoding="utf-8")
    with pytest.raises(ValueError, match="K_std=0"):
        load_reference(p)


def test_load_reference_rejects_missing_checkpoints(tmp_path: Path):
    """14'ten az ckpt -> ValueError."""
    fake = {
        "_meta": {"schema_version": "0.2.0"},
        "phases": {
            "PULL": {
                "checkpoints": [{
                    "id": "PULL_35", "T": 35,
                    "K_mean": 138.0, "K_std": 10.0,
                    "hip_shoulder_velocity_ratio_mean": 1.05,
                    "hip_shoulder_velocity_ratio_std": 0.25,
                }],
            },
            "DESCENT": {"checkpoints": []},
        },
    }
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(fake), encoding="utf-8")
    with pytest.raises(ValueError, match="Expected 14"):
        load_reference(p)


# ---- compute_z ----

def _sample(
    ckpt_id: str = "PULL_25",
    K: float = 158.0,
    velocity_ratio: float = 0.98,
    velocity_valid: bool = True,
) -> CheckpointSample:
    return CheckpointSample(
        id=ckpt_id,
        phase=Phase.PULL,
        T_target=25.0,
        T_observed=25.0,
        K=K,
        hip_shoulder_velocity_ratio=velocity_ratio,
        velocity_ratio_valid=velocity_valid,
        timestamp_ms=1000,
        interpolated=False,
    )


def _ref(
    ckpt_id: str = "PULL_25",
    K_mean: float = 158.0,
    K_std: float = 7.0,
    v_mean: float = 0.98,
    v_std: float = 0.18,
) -> CheckpointReference:
    return CheckpointReference(
        id=ckpt_id, T=25.0,
        K_mean=K_mean, K_std=K_std,
        hip_shoulder_velocity_ratio_mean=v_mean,
        hip_shoulder_velocity_ratio_std=v_std,
    )


def test_compute_z_zero_when_user_matches_mean():
    z = compute_z(_sample(), _ref())
    assert z.K_z == pytest.approx(0.0)
    assert z.velocity_ratio_z == pytest.approx(0.0)


def test_compute_z_known_negative_K():
    """K=148 vs mean=158 std=7 -> z = -10/7 ≈ -1.4286."""
    z = compute_z(_sample(K=148.0), _ref())
    assert z.K_z == pytest.approx(-10.0 / 7.0, abs=1e-6)


def test_compute_z_velocity_invalid_returns_none():
    """sample.velocity_ratio_valid=False -> velocity_ratio_z=None."""
    z = compute_z(_sample(velocity_valid=False), _ref())
    assert z.velocity_ratio_z is None
    # K kanali yine hesaplanir
    assert z.K_z == pytest.approx(0.0)


def test_compute_z_id_mismatch_raises():
    with pytest.raises(ValueError, match="mismatches"):
        compute_z(_sample(ckpt_id="PULL_25"), _ref(ckpt_id="DESCENT_25"))


def test_compute_z_returns_correct_id():
    z = compute_z(_sample(), _ref())
    assert z.checkpoint_id == "PULL_25"
    assert isinstance(z, ZScores)
