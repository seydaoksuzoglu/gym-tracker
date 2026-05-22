import pytest

from src.analysis.deadlift.scoring import (
    area_from_z, confidence_from_z, ui_level, overall_grade,
)


# area_from_z

@pytest.mark.parametrize("z,expected", [
    (0.0, 1), (0.99, 1), (-0.5, 1),
    (1.0, 2), (1.99, 2), (-1.5, 2),
    (2.0, 3), (2.99, 3),
    (3.0, 4), (3.99, 4),
    (4.0, 5), (10.0, 5), (-10.0, 5),
])
def test_area_from_z_boundaries(z, expected):
    assert area_from_z(z) == expected


# confidence_from_z

def test_confidence_zero_at_z_zero():
    assert confidence_from_z(0.0) == 0.0


def test_confidence_caps_at_one():
    assert confidence_from_z(4.0) == 1.0
    assert confidence_from_z(10.0) == 1.0
    assert confidence_from_z(-8.0) == 1.0


def test_confidence_linear_below_cap():
    assert confidence_from_z(2.0) == pytest.approx(0.5)
    assert confidence_from_z(1.0) == pytest.approx(0.25)


# ui_level

@pytest.mark.parametrize("c,expected", [
    (0.0, "silent"), (0.29, "silent"),
    (0.30, "yellow"), (0.59, "yellow"),
    (0.60, "red"), (1.0, "red"),
])
def test_ui_level_boundaries(c, expected):
    assert ui_level(c) == expected


# overall_grade

def test_overall_grade_empty_is_perfect():
    assert overall_grade([]) == "perfect"


@pytest.mark.parametrize("areas,grade", [
    ([1, 1, 1], "perfect"),
    ([1, 2], "good"),
    ([3, 1], "needs_attention"),
    ([4, 2, 1], "form_issue"),
    ([5], "severe"),
    ([2, 5, 1], "severe"),
])
def test_overall_grade_takes_worst(areas, grade):
    assert overall_grade(areas) == grade
