"""
Squat pipeline regression snapshot test

Refactor sırasında çalıştırılır. Mevcut squat sisteminin bilinen bir video üzerindeki
çıktısını "altın standart" olarak dondurur. Çıktı bozulduysa fail eder.

Snapshot oluşturmak için (ilk koşum):
    $env:UPDATE_SNAPSHOT="1"; python -m pytest tests/smoke/test_squat_still_works.py -s

Doğrulama için (sonraki koşumlar):
    python -m pytest tests/smoke/test_squat_still_works.py

"""

import json
import os
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import pytest
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Repo root'u path'e ekle
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.squat.squat_analyzer import LiveSquatAnalyzer
from src.sources.video import video_frames

# Sabitler
FIXTURE_VIDEO = REPO_ROOT / "tests" / "fixtures" / "video_002.mp4"
EXPECTED_OUTPUT = REPO_ROOT / "tests" / "fixtures" / "expected_output_video_002.json"
MODEL_PATH = REPO_ROOT / "models" / "pose_landmarker_full.task"

# Toleranslar (Kabul Kriteri: faz geçişleri +-50)
TS_TOLERANCE_MS = 50

def create_landmarker(model_path: Path):
    """run_pose.py'deki create_landmarker'ın aynısı - UI bağımlılığı yok."""
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.65,
        min_pose_presence_confidence=0.65,
        min_tracking_confidence=0.70,
    )
    return vision.PoseLandmarker.create_from_options(options)

def run_squat_pipeline_headless(video_path: Path, model_path: Path) -> dict:
    """
    UI'sız squat pipeline'ı koşturur. run_pose.py'nin minimal versiyonu.
    Sadece analyzer çıktılarını toplar — cv2.imshow, writer, draw yok.

    Returns:
        {
          "video": str,
          "total_reps": int,
          "reps": [{"rep_count", "ts_ms", "has_error", "error_labels"}, ...],
          "phase_transitions": [{"ts_ms", "state"}, ...]
        }
    """
    landmarker = create_landmarker(model_path)
    analyzer = LiveSquatAnalyzer()

    reps = []
    phase_transitions = []
    last_state = None

    try:
        for frame_bgr, ts in video_frames(str(video_path)):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, ts)

            analysis = analyzer.analyze(result, ts)
            if analysis is None:
                continue

            # Faz geçişlerini yakala (sadece state değişimi anlarını kaydet)
            if analysis.state != last_state:
                phase_transitions.append({"ts_ms": ts, "state": analysis.state})
                last_state = analysis.state

            # Tamamlanan rep'leri kaydet
            if analysis.rep_feedback is not None:
                rf = analysis.rep_feedback
                reps.append({
                    "rep_count": rf.rep_count,
                    "ts_ms": ts,
                    "has_error": rf.has_error,
                    "error_labels": sorted(rf.error_labels),
                })
    finally:
        landmarker.close()

    return {
        "video": video_path.name,
        "total_reps": len(reps),
        "reps": reps,
        "phase_transitions": phase_transitions,
    }

def assert_reps_match(actual: list, expected: list):
    """Rep listesi tam eşleşmeli. ts_ms ±50ms tolerans, geri kalanı tam."""
    assert len(actual) == len(expected), (
        f"Rep sayısı uyuşmuyor: actual={len(actual)}, expected={len(expected)}"
    )
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert a["rep_count"] == e["rep_count"], (
            f"Rep {i}: rep_count uyuşmuyor — actual={a['rep_count']}, expected={e['rep_count']}"
        )
        assert a["has_error"] == e["has_error"], (
            f"Rep {i}: has_error uyuşmuyor — actual={a['has_error']}, expected={e['has_error']}"
        )
        assert a["error_labels"] == e["error_labels"], (
            f"Rep {i}: error_labels uyuşmuyor — "
            f"actual={a['error_labels']}, expected={e['error_labels']}"
        )
        ts_delta = abs(a["ts_ms"] - e["ts_ms"])
        assert ts_delta <= TS_TOLERANCE_MS, (
            f"Rep {i}: ts_ms tolerans dışı ({ts_delta}ms > {TS_TOLERANCE_MS}ms) — "
            f"actual={a['ts_ms']}, expected={e['ts_ms']}"
        )


def assert_phases_match(actual: list, expected: list):
    """Faz geçişleri sırası tam, ts_ms ±50ms tolerans."""
    assert len(actual) == len(expected), (
        f"Faz geçiş sayısı uyuşmuyor: actual={len(actual)}, expected={len(expected)}"
    )
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert a["state"] == e["state"], (
            f"Faz geçiş {i}: state uyuşmuyor — actual={a['state']}, expected={e['state']}"
        )
        ts_delta = abs(a["ts_ms"] - e["ts_ms"])
        assert ts_delta <= TS_TOLERANCE_MS, (
            f"Faz geçiş {i} ({a['state']}): ts_ms tolerans dışı "
            f"({ts_delta}ms > {TS_TOLERANCE_MS}ms) — "
            f"actual={a['ts_ms']}, expected={e['ts_ms']}"
        )


def test_squat_pipeline_unchanged():
    """Mevcut squat pipeline'ının çıktısı snapshot ile birebir uyuşmalı."""
    if not FIXTURE_VIDEO.exists():
        pytest.skip(f"Fixture video bulunamadı: {FIXTURE_VIDEO}")
    if not MODEL_PATH.exists():
        pytest.skip(f"Pose model bulunamadı: {MODEL_PATH}")

    output = run_squat_pipeline_headless(FIXTURE_VIDEO, MODEL_PATH)

    # Snapshot güncelleme modu
    if os.environ.get("UPDATE_SNAPSHOT") == "1":
        EXPECTED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(EXPECTED_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n[snapshot] {EXPECTED_OUTPUT} yazıldı.")
        print(f"[snapshot] total_reps={output['total_reps']}, "
              f"phase_transitions={len(output['phase_transitions'])}")
        return

    # Doğrulama modu
    if not EXPECTED_OUTPUT.exists():
        pytest.fail(
            f"Snapshot bulunamadı: {EXPECTED_OUTPUT}\n"
            f"İlk kez koşturuyorsan: UPDATE_SNAPSHOT=1 pytest {Path(__file__).name}"
        )

    with open(EXPECTED_OUTPUT, "r", encoding="utf-8") as f:
        expected = json.load(f)

    # Kritik: total_reps tam eşleşmeli
    assert output["total_reps"] == expected["total_reps"], (
        f"Toplam rep sayısı değişti! actual={output['total_reps']}, "
        f"expected={expected['total_reps']}"
    )

    assert_reps_match(output["reps"], expected["reps"])
    assert_phases_match(output["phase_transitions"], expected["phase_transitions"])
