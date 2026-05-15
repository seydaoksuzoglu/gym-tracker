"""Geometry helper'lari icin unit testler."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import pytest
from src.common.geometry import angle, angle_to_vertical, distance, choose_best_side


def test_distance_basic():
    assert distance((0, 0), (3, 4)) == pytest.approx(5.0)
    assert distance((1, 1), (1, 1)) == 0.0


def test_angle_right_angle():
    assert angle((1, 0), (0, 0), (0, 1)) == pytest.approx(90.0)


def test_angle_straight():
    assert angle((0, 0), (1, 0), (2, 0)) == pytest.approx(180.0)


def test_angle_zero_vector_returns_zero():
    assert angle((0, 0), (0, 0), (1, 1)) == 0.0


def test_angle_to_vertical_straight_up():
    assert angle_to_vertical(top=(0, 0), bottom=(0, 1)) == pytest.approx(0.0)


def test_angle_to_vertical_horizontal():
    assert angle_to_vertical(top=(1, 0), bottom=(0, 0)) == pytest.approx(90.0)


class _FakeLandmark:
    def __init__(self, visibility):
        self.visibility = visibility


def test_choose_best_side_picks_higher_visibility():
    landmarks = [None] * 33
    landmarks[11] = _FakeLandmark(0.9)
    landmarks[23] = _FakeLandmark(0.9)
    landmarks[12] = _FakeLandmark(0.4)
    landmarks[24] = _FakeLandmark(0.4)
    name, _, vis = choose_best_side(
        landmarks,
        left_map={"shoulder": 11, "hip": 23},
        right_map={"shoulder": 12, "hip": 24},
    )
    assert name == "left"
    assert vis == pytest.approx(0.9)


def test_choose_best_side_tie_prefers_left():
    landmarks = [None] * 33
    landmarks[11] = _FakeLandmark(0.5)
    landmarks[12] = _FakeLandmark(0.5)
    name, _, _ = choose_best_side(
        landmarks, {"shoulder": 11}, {"shoulder": 12}
    )
    assert name == "left"
