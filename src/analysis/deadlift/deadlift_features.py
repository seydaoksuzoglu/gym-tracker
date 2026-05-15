"""
Geometri cikarimi. SADECE geometri - is mantigi / view tespiti / esik karsilastirma YOK.
Filtrelenmis (+ kalibrasyondan sonra cagrilan) landmark frame'i alir,
T (govde-dikey), K (diz), back_angle, hip.y cikarir.
"""
from dataclasses import dataclass

from src.analysis.deadlift.landmarks import LEFT, RIGHT, LandmarkFrame
from src.common.geometry import angle, angle_to_vertical, choose_best_side


@dataclass(frozen=True)
class DeadliftFeatures:
    side: str          # "left" / "right"
    T: float           # govde-dikey aci (derece). SETUP'ta yuksek, LOCKOUT'ta ~0.
    K: float           # diz acisi (derece). hip-knee-ankle.
    back_angle: float  # omuz-kalca hattinin yatayla acisi (derece).
    hip_y: float       # filtrelenmis kalca y koordinati.
    avg_visibility: float
    valid: bool


def extract_deadlift_features(frame: LandmarkFrame) -> DeadliftFeatures:
    side_name, side_map, avg_vis = choose_best_side(
        frame.landmarks, LEFT, RIGHT,
    )

    shoulder = frame.get(side_map["shoulder"])
    hip = frame.get(side_map["hip"])
    knee = frame.get(side_map["knee"])
    ankle = frame.get(side_map["ankle"])

    if not all(p.valid for p in (shoulder, hip, knee, ankle)):
        return DeadliftFeatures(
            side=side_name,
            T=0.0, K=0.0, back_angle=0.0,
            hip_y=hip.y if hip.valid else -1.0,
            avg_visibility=avg_vis,
            valid=False,
        )

    s_xy = shoulder.as_xy()
    h_xy = hip.as_xy()
    k_xy = knee.as_xy()
    a_xy = ankle.as_xy()

    T = angle_to_vertical(top=s_xy, bottom=h_xy)
    K = angle(h_xy, k_xy, a_xy)
    back_angle = 90.0 - T

    return DeadliftFeatures(
        side=side_name,
        T=T,
        K=K,
        back_angle=back_angle,
        hip_y=h_xy[1],
        avg_visibility=avg_vis,
        valid=True,
    )
