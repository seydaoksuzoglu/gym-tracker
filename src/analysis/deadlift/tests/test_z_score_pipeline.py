# src/analysis/deadlift/tests/test_z_score_pipeline.py
"""
Mini-pipeline testi: CheckpointCollector + compute_z + rep boundary glue.
LiveDeadliftAnalyzer'in tam mock'unu yapmak yerine, glue mantigini
bagimsiz dogrular (analyzer e2e testi ayri).
"""
from typing import List

from src.analysis.deadlift.checkpoints import CheckpointCollector
from src.analysis.deadlift.deadlift_features import DeadliftFeatures
from src.analysis.deadlift.phase_detector import Phase
from src.analysis.deadlift.z_score import ZScores, compute_z, load_reference


def _feat(T, K=160.0, hip_y=0.50, shoulder_y=0.25):
    return DeadliftFeatures(
        side="left", T=T, K=K, back_angle=90 - T,
        hip_y=hip_y, shoulder_y=shoulder_y,
        avg_visibility=0.9, valid=True,
    )


def test_full_pull_produces_seven_z_scores():
    """PULL boyunca T 36->4 -> 7 ckpt -> 7 ZScores."""
    refs = load_reference()
    c = CheckpointCollector()
    z_buffer: List[ZScores] = []

    for i, T in enumerate(36.0 - 0.5 * j for j in range(70)):
        ts = i * 33
        hip = 0.50 - 0.005 * i
        emitted = c.update(Phase.PULL, _feat(T=T, hip_y=hip,
                                              shoulder_y=hip - 0.25), ts)
        for s in emitted:
            z_buffer.append(compute_z(s, refs[s.id]))

    assert len(z_buffer) == 7
    assert {z.checkpoint_id for z in z_buffer} == {
        "PULL_35", "PULL_30", "PULL_25", "PULL_20",
        "PULL_15", "PULL_10", "PULL_5"
    }
    # K=160 sabit; mean 138-179 araliginda degisiyor -> z hep finite ve farkli
    K_zs = [z.K_z for z in z_buffer]
    assert all(abs(z) < 10 for z in K_zs)  # makul aralik


def test_rep_boundary_isolates_z_scores():
    """Iki ardisik rep -> her birinin kendi z-buffer'i."""
    refs = load_reference()
    c = CheckpointCollector()
    rep1_z: List[ZScores] = []
    rep2_z: List[ZScores] = []

    def feed(phase: Phase, T_series, ts_start, buf):
        for i, T in enumerate(T_series):
            ts = ts_start + i * 33
            hip = 0.50 - 0.005 * (i + ts_start // 33)
            emitted = c.update(phase, _feat(T=T, hip_y=hip,
                                             shoulder_y=hip - 0.25), ts)
            for s in emitted:
                buf.append(compute_z(s, refs[s.id]))

    # Rep 1
    feed(Phase.PULL, [36.0 - 0.5 * i for i in range(70)], 0, rep1_z)
    feed(Phase.LOCKOUT, [3.0, 3.0], 5000, rep1_z)
    feed(Phase.DESCENT, [4.0 + 0.5 * i for i in range(70)], 6000, rep1_z)

    # Rep boundary (analyzer'in yaptigi sey)
    c.start_new_rep()

    # Rep 2
    feed(Phase.SETUP, [40.0, 40.0], 12000, rep2_z)
    feed(Phase.PULL, [36.0 - 0.5 * i for i in range(70)], 13000, rep2_z)

    rep1_ids = {z.checkpoint_id for z in rep1_z}
    rep2_ids = {z.checkpoint_id for z in rep2_z}

    # Rep 1: 7 PULL + 7 DESCENT
    assert len(rep1_ids) == 14
    # Rep 2: yine 7 PULL (PULL_35 vb. start_new_rep sonrasi tekrar yakalandi)
    assert "PULL_35" in rep2_ids
    assert len(rep2_z) == 7
