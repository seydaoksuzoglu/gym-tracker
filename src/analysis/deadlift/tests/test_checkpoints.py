"""
CheckpointCollector birim testleri.

Sentetik DeadliftFeatures dizileri uzerinde calisir. Gercek video gerekmez.
"""
from typing import List

import pytest

from src.analysis.deadlift.checkpoints import (
    CHECKPOINT_T_DEGREES,
    CheckpointCollector,
    CheckpointSample,
    FAST_TRANSITION_DEG,
    SHOULDER_VELOCITY_EPS,
    TOLERANCE_DEG,
)
from src.analysis.deadlift.deadlift_features import DeadliftFeatures
from src.analysis.deadlift.phase_detector import Phase


# ---- yardimcilar ----

def _feat(
    T: float,
    K: float = 160.0,
    hip_y: float = 0.50,
    shoulder_y: float = 0.25,
    valid: bool = True,
) -> DeadliftFeatures:
    return DeadliftFeatures(
        side="left", T=T, K=K, back_angle=90 - T,
        hip_y=hip_y, shoulder_y=shoulder_y,
        avg_visibility=0.9, valid=valid,
    )


def _drive(
    collector: CheckpointCollector,
    phase: Phase,
    T_series: List[float],
    K_series: List[float] = None,
    hip_y_series: List[float] = None,
    shoulder_y_series: List[float] = None,
    ts_step: int = 33,
    ts_start: int = 0,
) -> List[CheckpointSample]:
    """T_series boyunca update cagir, tum emit'leri biriktir."""
    n = len(T_series)
    if K_series is None:
        K_series = [160.0] * n
    if hip_y_series is None:
        hip_y_series = [0.50 - 0.005 * i for i in range(n)]  # hip rises (y decreases)
    if shoulder_y_series is None:
        shoulder_y_series = [v - 0.25 for v in hip_y_series]  # ayni hizla yukseliyor
    out: List[CheckpointSample] = []
    for i, T in enumerate(T_series):
        ts = ts_start + i * ts_step
        f = _feat(T=T, K=K_series[i],
                  hip_y=hip_y_series[i], shoulder_y=shoulder_y_series[i])
        out.extend(collector.update(phase, f, ts))
    return out


# ---- testler ----

def test_setup_phase_emits_nothing():
    c = CheckpointCollector()
    out = _drive(c, Phase.SETUP, T_series=[40, 35, 30, 25])
    assert out == []
    assert c.collected() == []


def test_lockout_phase_emits_nothing():
    c = CheckpointCollector()
    out = _drive(c, Phase.LOCKOUT, T_series=[5, 4, 3, 2])
    assert out == []


def test_pull_direct_match_all_seven_checkpoints():
    """T 36 -> 3 derece arasi yumusak iniste 7 ckpt'in hepsi yakalanmali."""
    c = CheckpointCollector()
    T_series = [36.0 - 0.5 * i for i in range(70)]  # 36.0, 35.5, ..., 1.5
    out = _drive(c, Phase.PULL, T_series=T_series)
    ids = sorted({s.id for s in out})
    expected = sorted(f"PULL_{t}" for t in CHECKPOINT_T_DEGREES)
    assert ids == expected
    # Hicbiri interpolate edilmemis olmali (adim 0.5 deg)
    assert all(not s.interpolated for s in out)


def test_multi_trigger_closest_wins():
    """T ayni ckpt bandinda birden fazla frame'de gezinirse, target'a en yakin frame kaydedilir."""
    c = CheckpointCollector()
    # T=20 ckpt'i icin tolerans = 1.5; bant [18.5, 21.5]
    # Sirayla 20.8, 19.7, 20.05, 20.4 ... sonra 18.0'a duserek banttan cik
    T_series = [22.0, 20.8, 19.7, 20.05, 20.4, 18.0, 16.0]
    out = _drive(c, Phase.PULL, T_series=T_series)
    pull_20 = [s for s in out if s.id == "PULL_20"]
    assert len(pull_20) == 1
    # target=20'ye en yakin gozlem 20.05'ti
    assert pull_20[0].T_observed == pytest.approx(20.05, abs=1e-6)
    assert not pull_20[0].interpolated


def test_fast_transition_interpolates_missed_checkpoint():
    """prev_T=24, curr_T=19 (delta=5 > 3) -> T=20 ckpt'i interpole edilmeli."""
    c = CheckpointCollector()
    # 30'dan girip 27 -> 24 -> 19 (sicrama burada) -> 16 -> ...
    T_series = [30.0, 27.0, 24.0, 19.0, 16.0, 13.0]
    out = _drive(c, Phase.PULL, T_series=T_series)
    pull_20 = [s for s in out if s.id == "PULL_20"]
    assert len(pull_20) == 1
    assert pull_20[0].interpolated is True
    # f = (20 - 24) / (19 - 24) = 0.8 ; observed_T tam target degerine eslenir
    assert pull_20[0].T_observed == pytest.approx(20.0, abs=1e-6)


def test_fast_transition_skips_already_locked():
    """Daha onceki bir frame'de bant icinde locked olan ckpt interpolasyonla overwrite olmaz."""
    c = CheckpointCollector()
    # T=20'yi normal yoldan yakala, sonra hizli gecisle T=20'yi yeniden gec
    T_series = [22, 20.0, 17.0, 20.0, 14.0]  # 4. frame T=20 ama zaten locked
    out = _drive(c, Phase.PULL, T_series=T_series)
    pull_20 = [s for s in out if s.id == "PULL_20"]
    assert len(pull_20) == 1
    assert pull_20[0].interpolated is False


def test_pull_then_descent_separate_ids():
    """Ayni T degeri PULL ve DESCENT'te ayri ckpt id'lerine yazilir."""
    c = CheckpointCollector()
    # PULL: 36 -> 4
    _drive(c, Phase.PULL, T_series=[36.0 - 0.5 * i for i in range(70)])
    # LOCKOUT araliginda pending kalmasin (zaten cikti, kontrol icin)
    _drive(c, Phase.LOCKOUT, T_series=[3.0, 3.0], ts_start=10000)
    # DESCENT: 4 -> 36
    out_desc = _drive(c, Phase.DESCENT,
                      T_series=[4.0 + 0.5 * i for i in range(70)], ts_start=20000)
    desc_ids = {s.id for s in out_desc}
    expected = {f"DESCENT_{t}" for t in CHECKPOINT_T_DEGREES}
    assert desc_ids == expected
    # PULL ve DESCENT id seti birbirine girmemeli
    all_ids = {s.id for s in c.collected()}
    assert all_ids == {f"PULL_{t}" for t in CHECKPOINT_T_DEGREES} | expected


def test_velocity_ratio_one_when_hip_and_shoulder_move_together():
    """hip ve omuz ayni hizla yukseliyorsa ratio = 1.0."""
    c = CheckpointCollector()
    # 0.005/frame yukselis (negatif y delta cunku y asagi)
    T_series = [36.0 - 0.5 * i for i in range(20)]
    out = _drive(c, Phase.PULL, T_series=T_series)
    pull_35 = next(s for s in out if s.id == "PULL_35")
    assert pull_35.velocity_ratio_valid
    assert pull_35.hip_shoulder_velocity_ratio == pytest.approx(1.0, abs=1e-6)


def test_velocity_ratio_invalid_when_shoulder_static():
    """Omuz hiz epsilon altindaysa velocity_ratio_valid=False."""
    c = CheckpointCollector()
    n = 20
    T_series = [36.0 - 0.5 * i for i in range(n)]
    hip = [0.50 - 0.005 * i for i in range(n)]
    shoulder = [0.25] * n  # sabit
    out = _drive(c, Phase.PULL, T_series=T_series,
                 hip_y_series=hip, shoulder_y_series=shoulder)
    pull_35 = next(s for s in out if s.id == "PULL_35")
    assert pull_35.velocity_ratio_valid is False
    assert pull_35.hip_shoulder_velocity_ratio == 0.0


def test_reset_clears_state():
    c = CheckpointCollector()
    _drive(c, Phase.PULL, T_series=[36.0 - 0.5 * i for i in range(70)])
    assert len(c.collected()) == 7
    c.reset()
    assert c.collected() == []
    # reset sonrasi PULL_35'i tekrar yakalayabilmeli
    out2 = _drive(c, Phase.PULL, T_series=[36.0, 35.0, 33.0], ts_start=50000)
    assert any(s.id == "PULL_35" for s in out2)


def test_invalid_features_ignored():
    c = CheckpointCollector()
    out = _drive(c, Phase.PULL,
                 T_series=[35.0, 35.0, 35.0],
                 K_series=[160.0] * 3,
                 hip_y_series=[0.5, 0.5, 0.5],
                 shoulder_y_series=[0.25, 0.25, 0.25])
    # Bant icinde kaldi, henuz finalize olmadi; emit yok
    assert out == []
    # Simdi invalid frame at - state degismemeli
    inv = c.update(Phase.PULL, _feat(T=0.0, valid=False), ts_ms=10000)
    assert inv == []


def test_checkpoint_id_format():
    c = CheckpointCollector()
    out = _drive(c, Phase.PULL, T_series=[36.0 - 0.5 * i for i in range(20)])
    sample = next(s for s in out if abs(s.T_target - 35) < 0.01)
    assert sample.id == "PULL_35"
    assert sample.phase == Phase.PULL


def test_phase_change_flushes_pending():
    """PULL bandindayken DESCENT'e gecilirse pending PULL ckpt finalize edilir."""
    c = CheckpointCollector()
    # PULL_35 bandina gir (35.0), sonra fazi DESCENT'e cevir - banttan dogal cikis yok
    samples_pull = _drive(c, Phase.PULL,
                          T_series=[35.5, 35.0, 34.8], ts_start=0)
    assert samples_pull == []  # henuz finalize degil

    # DESCENT'e gec - pending olan PULL_35 flush edilmeli
    samples_desc = _drive(c, Phase.DESCENT, T_series=[5.0, 5.0], ts_start=10000)
    flushed_ids = {s.id for s in samples_desc}
    assert "PULL_35" in flushed_ids
    
def test_start_new_rep_clears_locked_but_keeps_prev():
    """start_new_rep _locked ve _pending'i siler ama _prev'i tutar."""
    c = CheckpointCollector()
    _drive(c, Phase.PULL, T_series=[36.0 - 0.5 * i for i in range(70)])
    assert len(c.collected()) == 7
    prev_before = c._prev

    c.start_new_rep()
    assert c.collected() == []
    assert c._prev is prev_before  # state surekliliği

    # Sonraki frame ckpt'leri yeniden yakalayabilmeli
    out = _drive(c, Phase.PULL,
                 T_series=[36.0 - 0.5 * i for i in range(20)],
                 ts_start=100000)
    assert any(s.id == "PULL_35" for s in out)
