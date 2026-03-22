import math
from src.analysis.squat_features import SquatFeatures


def _angle(a, b, c):
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


def extract_squat_features_yolo(yf) -> SquatFeatures:
    p = yf.points
    c = yf.confs

    ls = p["left_shoulder"]
    rs = p["right_shoulder"]
    lh = p["left_hip"]
    rh = p["right_hip"]
    lk = p["left_knee"]
    rk = p["right_knee"]
    la = p["left_ankle"]
    ra = p["right_ankle"]

    left_conf = (c["left_shoulder"] + c["left_hip"] + c["left_knee"] + c["left_ankle"]) / 4.0
    right_conf = (c["right_shoulder"] + c["right_hip"] + c["right_knee"] + c["right_ankle"]) / 4.0

    if left_conf >= right_conf:
        side = "left"
        shoulder, hip, knee, ankle = ls, lh, lk, la
        avg_vis = left_conf
    else:
        side = "right"
        shoulder, hip, knee, ankle = rs, rh, rk, ra
        avg_vis = right_conf

    left_knee_angle = _angle(lh, lk, la)
    right_knee_angle = _angle(rh, rk, ra)

    left_hip_angle = _angle(ls, lh, lk)
    right_hip_angle = _angle(rs, rh, rk)

    knee_angle = left_knee_angle if side == "left" else right_knee_angle
    hip_angle = left_hip_angle if side == "left" else right_hip_angle

    shoulder_span = abs(ls[0] - rs[0])
    hip_span = abs(lh[0] - rh[0])

    shoulder_center = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
    hip_center = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
    torso_height = abs(hip_center[1] - shoulder_center[1])

    # Top-down benzeri kaba diz hizası metriği
    hip_width = max(hip_span, 1e-6)
    left_valgus = max(0.0, lk[0] - la[0]) / hip_width
    right_valgus = max(0.0, ra[0] - rk[0]) / hip_width
    knee_valgus_offset = max(left_valgus, right_valgus)

    if left_valgus > right_valgus:
        valgus_side = "left"
    elif right_valgus > left_valgus:
        valgus_side = "right"
    else:
        valgus_side = "both"

    knee_asymmetry = abs(lk[1] - rk[1]) / max(torso_height, 1e-6)

    frontal_width = (shoulder_span + hip_span) / 2.0
    view_ratio = frontal_width / max(torso_height, 1e-6)
    view_label = "side" if view_ratio < 0.25 else "front"

    return SquatFeatures(
        side=side,
        knee_angle=knee_angle,
        hip_angle=hip_angle,
        trunk_angle=0.0,
        shin_angle=0.0,
        hip_y=hip[1],
        knee_y=knee[1],
        ankle_y=ankle[1],
        foot_length=1.0,
        heel_lift_ratio=0.0,   # YOLO26 COCO17'de heel yok
        shoulder_span=shoulder_span,
        hip_span=hip_span,
        torso_height=torso_height,
        avg_visibility=avg_vis,
        hip_below_knee=(hip[1] > knee[1]),
        valid=yf.valid,
        left_knee_angle=left_knee_angle,
        right_knee_angle=right_knee_angle,
        left_hip_angle=left_hip_angle,
        right_hip_angle=right_hip_angle,
        knee_asymmetry=knee_asymmetry,
        knee_valgus_offset=knee_valgus_offset,
        valgus_side=valgus_side,
        view_label=view_label,
    )