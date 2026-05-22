"""
Tam pipeline'i video uzerinde calistirir, manuel etiketle karsilastirir.
Testi koş: pytest src/analysis/deadlift/tests/test_e2e_deadlift_001.py -v -s
"""
import json
from pathlib import Path

import cv2
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
MODEL_PATH = REPO_ROOT / "models" / "pose_landmarker_full.task"

VIDEO_NAME = "deadlift_002.mp4"
EXPECTED_JSON = FIXTURE_DIR / "deadlift_002_phases.json"
TOLERANCE_MS = 500


@pytest.mark.skipif(
    not (FIXTURE_DIR / VIDEO_NAME).exists() or not EXPECTED_JSON.exists(),
    reason="Video veya etiket dosyasi yok",
)
def test_phase_transitions_match_manual_labels():
    from src.analysis.deadlift.deadlift_analyzer import LiveDeadliftAnalyzer

    expected = json.loads(EXPECTED_JSON.read_text(encoding="utf-8"))
    analyzer = LiveDeadliftAnalyzer(model_path=str(MODEL_PATH))

    detected_transitions = []
    cap = cv2.VideoCapture(str(FIXTURE_DIR / VIDEO_NAME))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    prev_phase = None

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            ts_ms = int(frame_idx * 1000 / fps)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            analysis = analyzer.analyze(frame_rgb, ts_ms)
            if not analysis.calibrating and prev_phase != analysis.phase:
                if prev_phase is not None:
                    detected_transitions.append({
                        "from": prev_phase,
                        "to": analysis.phase,
                        "ts_ms": ts_ms,
                    })
                prev_phase = analysis.phase
            frame_idx += 1
    finally:
        cap.release()
        analyzer.close()

    # 1. Rep sayisi
    completed_reps = sum(
        1 for t in detected_transitions
        if t["from"] == "descent" and t["to"] in ("setup", "pull")
        and any(
            t2["to"] == "lockout" for t2 in detected_transitions
            if t2["ts_ms"] < t["ts_ms"]
        )
    )
    # Daha basit: rep_counter zaten saydi
    # Burada analyzer.rep_counter.rep_count'a guvenebiliriz
    # (test'in son halinde analyzer.rep_counter.rep_count okunur)

    # Tanı için yan yana yazdır
    expected_trans = expected["phase_transitions"]
    print("\n--- BEKLENEN (manuel) vs TESPIT (sistem) ---")
    max_len = max(len(expected_trans), len(detected_transitions))
    for i in range(max_len):
        e = expected_trans[i] if i < len(expected_trans) else None
        d = detected_transitions[i] if i < len(detected_transitions) else None
        e_str = f"{e['from']:>7}->{e['to']:<7}@{e['ts_ms']:>5}ms" if e else "---"
        d_str = f"{d['from']:>7}->{d['to']:<7}@{d['ts_ms']:>5}ms" if d else "---"
        diff = ""
        if e and d:
            dt = d['ts_ms'] - e['ts_ms']
            diff = f"  (+{dt}ms)" if dt >= 0 else f"  ({dt}ms)"
        print(f"  {i+1:2d}.  {e_str}    |    {d_str}{diff}")
    print()

    assert len(detected_transitions) == len(expected_trans), (
        f"Gecis sayisi farkli: beklenen={len(expected_trans)}, "
        f"tespit={len(detected_transitions)}"
    )

