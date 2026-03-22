from dataclasses import dataclass
from typing import Dict, Optional, Tuple


YOLO_LEFT = {
    "shoulder": 5,
    "hip": 11,
    "knee": 13,
    "ankle": 15,
}

YOLO_RIGHT = {
    "shoulder": 6,
    "hip": 12,
    "knee": 14,
    "ankle": 16,
}


@dataclass
class YoloPoseFrame:
    points: Dict[str, Tuple[float, float]]
    confs: Dict[str, float]
    valid: bool


def extract_yolo_pose_frame(result, conf_thresh: float = 0.30) -> Optional[YoloPoseFrame]:
    if result is None or result.keypoints is None:
        return None

    if result.keypoints.xyn is None or result.keypoints.conf is None:
        return None

    xyn = result.keypoints.xyn.cpu().numpy()   # (N, 17, 2)
    conf = result.keypoints.conf.cpu().numpy() # (N, 17)

    if len(xyn) == 0:
        return None

    # Şimdilik ilk kişiyi al
    kp = xyn[0]
    cf = conf[0]

    def get_pt(idx: int):
        return (float(kp[idx][0]), float(kp[idx][1]))

    def get_cf(idx: int):
        return float(cf[idx])

    points = {
        "left_shoulder": get_pt(YOLO_LEFT["shoulder"]),
        "right_shoulder": get_pt(YOLO_RIGHT["shoulder"]),
        "left_hip": get_pt(YOLO_LEFT["hip"]),
        "right_hip": get_pt(YOLO_RIGHT["hip"]),
        "left_knee": get_pt(YOLO_LEFT["knee"]),
        "right_knee": get_pt(YOLO_RIGHT["knee"]),
        "left_ankle": get_pt(YOLO_LEFT["ankle"]),
        "right_ankle": get_pt(YOLO_RIGHT["ankle"]),
    }

    confs = {
        "left_shoulder": get_cf(YOLO_LEFT["shoulder"]),
        "right_shoulder": get_cf(YOLO_RIGHT["shoulder"]),
        "left_hip": get_cf(YOLO_LEFT["hip"]),
        "right_hip": get_cf(YOLO_RIGHT["hip"]),
        "left_knee": get_cf(YOLO_LEFT["knee"]),
        "right_knee": get_cf(YOLO_RIGHT["knee"]),
        "left_ankle": get_cf(YOLO_LEFT["ankle"]),
        "right_ankle": get_cf(YOLO_RIGHT["ankle"]),
    }

    valid = min(confs.values()) >= conf_thresh
    return YoloPoseFrame(points=points, confs=confs, valid=valid)