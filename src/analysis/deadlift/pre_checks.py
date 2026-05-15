"""
Pipeline basinda calisir. Side-view dogrulamasi.
View tespiti deadlift_features.py'ye SIZDIRILMAZ - bu kontrol burada kalir.

Not: CLAUDE.md plani 'shoulder_width * 0.30' diyordu ama 2D shoulder_width
side view'da kucuk projeksiyonlu (~0.01) cikiyor, mantik tersine donuyor.
torso_length daha kararli bir referans (gorunumlere gore degismiyor).
"""
from dataclasses import dataclass
from typing import Optional

from src.analysis.deadlift.calibration import StandingBaseline
from src.analysis.deadlift.landmarks import LEFT, RIGHT, LandmarkFrame


@dataclass(frozen=True)
class RejectedFrame:
    reason: str


SIDE_VIEW_RATIO_THRESHOLD = 0.20  # shoulder_x / torso_length


def check_side_view(
    frame: LandmarkFrame,
    baseline: Optional[StandingBaseline],
    ratio_threshold: float = SIDE_VIEW_RATIO_THRESHOLD,
) -> Optional[RejectedFrame]:
    """Side-view kontrolu.

    Kalibrasyon hazir degilse pas gecer.
    Iki omuz arasi x-mesafesi torso_length * ratio_threshold'dan buyukse
    front view kabul edilir -> reddet.
    """
    if baseline is None:
        return None  # Kalibrasyon oncesi pasif

    ls = frame.get(LEFT["shoulder"])
    rs = frame.get(RIGHT["shoulder"])
    if not (ls.valid and rs.valid):
        return RejectedFrame(reason="missing_shoulders")

    shoulder_x_distance = abs(ls.x - rs.x)
    threshold = baseline.torso_length * ratio_threshold

    if shoulder_x_distance > threshold:
        return RejectedFrame(reason="not_side_view")

    return None
