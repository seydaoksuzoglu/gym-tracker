import math
from dataclasses import dataclass
from typing import Tuple


# -----------------------------------------------------------
# MediaPipe Pose landmark indexleri
# -----------------------------------------------------------
LEFT = {
    "shoulder": 11,
    "hip": 23,
    "knee": 25,
    "ankle": 27,
    "heel": 29,
    "foot_index": 31,
}

RIGHT = {
    "shoulder": 12,
    "hip": 24,
    "knee": 26,
    "ankle": 28,
    "heel": 30,
    "foot_index": 32,
}


# -----------------------------------------------------------
# Squat feature dataclass
# -----------------------------------------------------------
@dataclass
class SquatFeatures:
    side: str
    knee_angle: float
    hip_angle: float
    trunk_angle: float
    shin_angle: float

    hip_y: float
    knee_y: float
    ankle_y: float

    foot_length: float
    heel_lift_ratio: float

    shoulder_span: float
    hip_span: float
    torso_height: float

    avg_visibility: float
    hip_below_knee: bool
    valid: bool

    left_knee_angle: float
    right_knee_angle: float
    left_hip_angle: float
    right_hip_angle: float

    knee_asymmetry: float
    knee_valgus_offset: float
    valgus_side: str

    view_label: str


# -----------------------------------------------------------
# Yardımcı fonksiyonlar
# -----------------------------------------------------------
def _point(landmarks, idx: int) -> Tuple[float, float]:
    lm = landmarks[idx]
    return (lm.x, lm.y)


def _visibility(landmarks, idx: int) -> float:
    lm = landmarks[idx]
    return float(getattr(lm, "visibility", 1.0))


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if mag_ba == 0 or mag_bc == 0:
        return 0.0

    cos_value = dot / (mag_ba * mag_bc)
    cos_value = max(-1.0, min(1.0, cos_value))
    return math.degrees(math.acos(cos_value))


def _angle_to_vertical(top: Tuple[float, float], bottom: Tuple[float, float]) -> float:
    v = (top[0] - bottom[0], top[1] - bottom[1])
    vertical = (0.0, -1.0)

    dot = v[0] * vertical[0] + v[1] * vertical[1]
    mag_v = math.sqrt(v[0] ** 2 + v[1] ** 2)

    if mag_v == 0:
        return 0.0

    cos_value = dot / mag_v
    cos_value = max(-1.0, min(1.0, cos_value))
    return math.degrees(math.acos(cos_value))


def _average_visibility(landmarks, side_map: dict) -> float:
    vis_values = [_visibility(landmarks, idx) for idx in side_map.values()]
    return sum(vis_values) / len(vis_values)


def _choose_best_side(landmarks) -> Tuple[str, dict, float]:
    left_vis = _average_visibility(landmarks, LEFT)
    right_vis = _average_visibility(landmarks, RIGHT)

    if left_vis >= right_vis:
        return "left", LEFT, left_vis
    return "right", RIGHT, right_vis


# -----------------------------------------------------------
# Ana feature çıkarma fonksiyonu
# -----------------------------------------------------------
def extract_squat_features(
    landmarks,
    min_visibility: float = 0.50,
    side_view_ratio_threshold: float = 0.45,
) -> SquatFeatures:
    """
    MediaPipe pose landmark listesinden squat için gerekli feature'ları çıkarır.
    """

    # Hangi taraf daha görünür
    side_name, side_map, avg_vis = _choose_best_side(landmarks)

    # Seçilen taraf
    shoulder = _point(landmarks, side_map["shoulder"])
    hip = _point(landmarks, side_map["hip"])
    knee = _point(landmarks, side_map["knee"])
    ankle = _point(landmarks, side_map["ankle"])
    heel = _point(landmarks, side_map["heel"])
    foot = _point(landmarks, side_map["foot_index"])

    # Sol taraf
    left_shoulder = _point(landmarks, LEFT["shoulder"])
    left_hip = _point(landmarks, LEFT["hip"])
    left_knee = _point(landmarks, LEFT["knee"])
    left_ankle = _point(landmarks, LEFT["ankle"])
    left_heel = _point(landmarks, LEFT["heel"])
    left_foot = _point(landmarks, LEFT["foot_index"])

    # Sağ taraf
    right_shoulder = _point(landmarks, RIGHT["shoulder"])
    right_hip = _point(landmarks, RIGHT["hip"])
    right_knee = _point(landmarks, RIGHT["knee"])
    right_ankle = _point(landmarks, RIGHT["ankle"])
    right_heel = _point(landmarks, RIGHT["heel"])
    right_foot = _point(landmarks, RIGHT["foot_index"])

    # Ana açılar
    knee_angle = _angle(hip, knee, ankle)
    hip_angle = _angle(shoulder, hip, knee)
    trunk_angle = _angle_to_vertical(shoulder, hip)
    shin_angle = _angle_to_vertical(knee, ankle)

    # Sol/sağ açılar
    left_knee_angle = _angle(left_hip, left_knee, left_ankle)
    right_knee_angle = _angle(right_hip, right_knee, right_ankle)

    left_hip_angle = _angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = _angle(right_shoulder, right_hip, right_knee)

    # Uzunluklar
    foot_length = max(_distance(heel, foot), 1e-6)

    # Topuk kalkma oranı
    heel_lift_ratio = max(0.0, foot[1] - heel[1]) / foot_length

    # Omuz ve kalça açıklığı
    shoulder_span = abs(left_shoulder[0] - right_shoulder[0])
    hip_span = abs(left_hip[0] - right_hip[0])

    # Torso yüksekliği
    shoulder_center = (
        (left_shoulder[0] + right_shoulder[0]) / 2.0,
        (left_shoulder[1] + right_shoulder[1]) / 2.0,
    )
    hip_center = (
        (left_hip[0] + right_hip[0]) / 2.0,
        (left_hip[1] + right_hip[1]) / 2.0,
    )
    torso_height = abs(hip_center[1] - shoulder_center[1])

    # Derinlik için kaba kontrol
    hip_below_knee = hip[1] > knee[1]

    # Görünürlük
    valid = avg_vis >= min_visibility

    # -------------------------------------------------------
    # Önden görünüm için diz feature'ları
    # -------------------------------------------------------
    stance_width = max(abs(left_ankle[0] - right_ankle[0]), 1e-6)

    # Diz asimetrisi: iki dizin düşey farkı
    knee_asymmetry = abs(left_knee[1] - right_knee[1]) / max(torso_height, 1e-6)

    # Diz içe kapanma için kaba 2D ölçü
    # Sol diz içe kapandıkça x olarak sağa yaklaşır
    # Sağ diz içe kapandıkça x olarak sola yaklaşır
    left_valgus = max(0.0, left_knee[0] - left_ankle[0]) / stance_width
    right_valgus = max(0.0, right_ankle[0] - right_knee[0]) / stance_width

    knee_valgus_offset = max(left_valgus, right_valgus)

    if left_valgus > right_valgus:
        valgus_side = "left"
    elif right_valgus > left_valgus:
        valgus_side = "right"
    else:
        valgus_side = "both"

    # -------------------------------------------------------
    # View tahmini
    # -------------------------------------------------------
    frontal_width = (shoulder_span + hip_span) / 2.0
    view_ratio = frontal_width / max(torso_height, 1e-6)
    view_label = "side" if view_ratio < side_view_ratio_threshold else "front"
    
    if view_label != "front":
        knee_valgus_offset = 0.0
        valgus_side = "none"
        knee_asymmetry = 0.0

    return SquatFeatures(
        side=side_name,
        knee_angle=knee_angle,
        hip_angle=hip_angle,
        trunk_angle=trunk_angle,
        shin_angle=shin_angle,
        hip_y=hip[1],
        knee_y=knee[1],
        ankle_y=ankle[1],
        foot_length=foot_length,
        heel_lift_ratio=heel_lift_ratio,
        shoulder_span=shoulder_span,
        hip_span=hip_span,
        torso_height=torso_height,
        avg_visibility=avg_vis,
        hip_below_knee=hip_below_knee,
        valid=valid,
        left_knee_angle=left_knee_angle,
        right_knee_angle=right_knee_angle,
        left_hip_angle=left_hip_angle,
        right_hip_angle=right_hip_angle,
        knee_asymmetry=knee_asymmetry,
        knee_valgus_offset=knee_valgus_offset,
        valgus_side=valgus_side,
        view_label=view_label,
    )