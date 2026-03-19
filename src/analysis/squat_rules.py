# src/analysis/squat_rules.py

from dataclasses import dataclass, field
from collections import deque
from typing import Optional, List
import math


# MediaPipe Pose landmark indexleri
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


@dataclass
class RuleConfig:
    # smoothing
    smooth_window: int = 5

    # rep state machine
    descend_threshold: float = 145.0
    up_threshold: float = 160.0
    min_rom_deg: float = 25.0
    cooldown_frames: int = 6

    # error thresholds
    half_depth_threshold: float = 110.0
    knee_lockout_threshold: float = 165.0
    hip_lockout_threshold: float = 150.0
    knee_diff_threshold: float = 18.0
    hip_diff_threshold: float = 20.0

    # debug / robustness
    min_valid_angle: float = 40.0
    max_valid_angle: float = 200.0

    forward_lean_threshold: float = 35.0
    live_depth_margin: float = 8.0
    live_lockout_margin: float = 10.0


@dataclass
class RepFeedback:
    rep_count: int
    has_error: bool
    error_labels: List[str] = field(default_factory=list)

    # debug amaçlı
    min_knee_angle: Optional[float] = None
    end_knee_angle: Optional[float] = None
    end_hip_angle: Optional[float] = None
    max_knee_angle_diff: Optional[float] = None
    max_hip_angle_diff: Optional[float] = None
    max_torso_lean_deg: Optional[float] = None

@dataclass
class LiveAnalysis:
    rep_feedback: Optional[RepFeedback] = None
    live_warnings: List[str] = field(default_factory=list)
    state: str = "UP"
    avg_knee_angle: Optional[float] = None
    avg_hip_angle: Optional[float] = None
    avg_torso_lean_deg: Optional[float] = None

def vector_sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]


def vector_norm(v):
    return math.sqrt(sum(x * x for x in v))


def angle_between_vectors(v1, v2):
    norm1 = vector_norm(v1)
    norm2 = vector_norm(v2)

    if norm1 == 0 or norm2 == 0:
        return None

    dot = sum(v1[i] * v2[i] for i in range(len(v1)))
    cos_theta = dot / (norm1 * norm2)
    cos_theta = max(-1.0, min(1.0, cos_theta))

    angle_rad = math.acos(cos_theta)
    return math.degrees(angle_rad)


def angle_3pt(a, b, c):
    ba = vector_sub(a, b)
    bc = vector_sub(c, b)
    return angle_between_vectors(ba, bc)


def safe_mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def safe_max(values):
    values = [v for v in values if v is not None]
    return max(values) if values else None


def safe_abs_diff(a, b):
    if a is None or b is None:
        return None
    return abs(a - b)


def clamp_valid_angle(angle, cfg: RuleConfig):
    if angle is None:
        return None
    if angle < cfg.min_valid_angle or angle > cfg.max_valid_angle:
        return None
    return angle


def _lm_to_point(lm, use_z=True):
    if lm is None:
        return None

    if use_z and hasattr(lm, "z"):
        return [float(lm.x), float(lm.y), float(lm.z)]
    return [float(lm.x), float(lm.y)]


def _get_pose_landmarks(result):
    pose_landmarks_list = getattr(result, "pose_landmarks", None)
    if not pose_landmarks_list:
        return None, None

    image_landmarks = pose_landmarks_list[0]

    world_landmarks = None
    pose_world_landmarks_list = getattr(result, "pose_world_landmarks", None)
    if pose_world_landmarks_list:
        world_landmarks = pose_world_landmarks_list[0]

    return image_landmarks, world_landmarks


def compute_live_frame_features(result, cfg: RuleConfig):
    image_landmarks, world_landmarks = _get_pose_landmarks(result)
    if image_landmarks is None:
        return None

    def world_or_image(idx):
        if world_landmarks is not None and len(world_landmarks) > idx:
            return _lm_to_point(world_landmarks[idx], use_z=True)
        if len(image_landmarks) > idx:
            return _lm_to_point(image_landmarks[idx], use_z=False)
        return None

    def image_only(idx):
        if len(image_landmarks) > idx:
            return _lm_to_point(image_landmarks[idx], use_z=False)
        return None

    l_shoulder = world_or_image(LEFT_SHOULDER)
    r_shoulder = world_or_image(RIGHT_SHOULDER)
    l_hip = world_or_image(LEFT_HIP)
    r_hip = world_or_image(RIGHT_HIP)
    l_knee = world_or_image(LEFT_KNEE)
    r_knee = world_or_image(RIGHT_KNEE)
    l_ankle = world_or_image(LEFT_ANKLE)
    r_ankle = world_or_image(RIGHT_ANKLE)

    l_shoulder_img = image_only(LEFT_SHOULDER)
    r_shoulder_img = image_only(RIGHT_SHOULDER)
    l_hip_img = image_only(LEFT_HIP)
    r_hip_img = image_only(RIGHT_HIP)

    left_knee_angle = None
    right_knee_angle = None
    left_hip_angle = None
    right_hip_angle = None
    left_torso_lean_deg = None
    right_torso_lean_deg = None

    if l_hip and l_knee and l_ankle:
        left_knee_angle = angle_3pt(l_hip, l_knee, l_ankle)

    if r_hip and r_knee and r_ankle:
        right_knee_angle = angle_3pt(r_hip, r_knee, r_ankle)

    if l_shoulder and l_hip and l_knee:
        left_hip_angle = angle_3pt(l_shoulder, l_hip, l_knee)

    if r_shoulder and r_hip and r_knee:
        right_hip_angle = angle_3pt(r_shoulder, r_hip, r_knee)

    if l_shoulder_img and l_hip_img:
        torso_vec = vector_sub(l_shoulder_img, l_hip_img)
        vertical_up = [0.0, -1.0]
        left_torso_lean_deg = angle_between_vectors(torso_vec, vertical_up)

    if r_shoulder_img and r_hip_img:
        torso_vec = vector_sub(r_shoulder_img, r_hip_img)
        vertical_up = [0.0, -1.0]
        right_torso_lean_deg = angle_between_vectors(torso_vec, vertical_up)

    left_knee_angle = clamp_valid_angle(left_knee_angle, cfg)
    right_knee_angle = clamp_valid_angle(right_knee_angle, cfg)
    left_hip_angle = clamp_valid_angle(left_hip_angle, cfg)
    right_hip_angle = clamp_valid_angle(right_hip_angle, cfg)

    knee_angle_diff = safe_abs_diff(left_knee_angle, right_knee_angle)
    hip_angle_diff = safe_abs_diff(left_hip_angle, right_hip_angle)

    return {
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "left_hip_angle": left_hip_angle,
        "right_hip_angle": right_hip_angle,
        "left_torso_lean_deg": left_torso_lean_deg,
        "right_torso_lean_deg": right_torso_lean_deg,
        "knee_angle_diff": knee_angle_diff,
        "hip_angle_diff": hip_angle_diff,
        "avg_knee_angle": safe_mean([left_knee_angle, right_knee_angle]),
        "avg_hip_angle": safe_mean([left_hip_angle, right_hip_angle]),
        "avg_torso_lean_deg": safe_mean([left_torso_lean_deg, right_torso_lean_deg]),
    }


class SquatRuleEngine:
    def __init__(self, config: RuleConfig | None = None):
        self.cfg = config or RuleConfig()
        self.reset()

    def reset(self):
        self.rep_count = 0
        self.state = "UP"
        self.cooldown = 0

        self.knee_hist = deque(maxlen=self.cfg.smooth_window)
        self.hip_hist = deque(maxlen=self.cfg.smooth_window)
        self.lean_hist = deque(maxlen=self.cfg.smooth_window)

        self.rep_metrics = None
        self.prev_knee_smooth = None
    
    def _compute_live_warnings(self, features, knee_smooth, hip_smooth, lean_smooth):
        warnings = []

        knee_diff = features.get("knee_angle_diff")
        hip_diff = features.get("hip_angle_diff")

        # 1) Sağ-sol dengesizlik
        if (
            (knee_diff is not None and knee_diff > self.cfg.knee_diff_threshold) or
            (hip_diff is not None and hip_diff > self.cfg.hip_diff_threshold)
        ):
            warnings.append("Dizleri dengeli it")

        # 2) Fazla öne eğilme
        if lean_smooth is not None and lean_smooth > self.cfg.forward_lean_threshold:
            warnings.append("Govdeyi daha dik tut")

        # 3) Derinlik uyarısı
        if self.state == "DOWN" and self.rep_metrics is not None and knee_smooth is not None:
            min_knee = self.rep_metrics.get("min_knee")

            coming_up = (
                self.prev_knee_smooth is not None and
                knee_smooth > self.prev_knee_smooth + 1.5
            )

            shallow_so_far = (
                min_knee is None or
                min_knee > (self.cfg.half_depth_threshold + self.cfg.live_depth_margin)
            )

            if coming_up and shallow_so_far:
                warnings.append("Biraz daha derin in")

        # 4) Lockout uyarısı
        if self.state == "DOWN" and knee_smooth is not None:
            coming_up = (
                self.prev_knee_smooth is not None and
                knee_smooth > self.prev_knee_smooth + 1.0
            )

            near_top = knee_smooth > (self.cfg.up_threshold - self.cfg.live_lockout_margin)

            knee_not_locked = (
                knee_smooth is not None and
                knee_smooth < self.cfg.knee_lockout_threshold
            )

            hip_not_locked = (
                hip_smooth is not None and
                hip_smooth < self.cfg.hip_lockout_threshold
            )

            if coming_up and near_top and (knee_not_locked or hip_not_locked):
                warnings.append("Ustte tam kilitle")

        return list(dict.fromkeys(warnings))

    def _start_rep_metrics(self, knee_smooth, hip_smooth, lean_smooth):
        self.rep_metrics = {
            "start_knee": knee_smooth,
            "start_hip": hip_smooth,
            "min_knee": knee_smooth,
            "min_hip": hip_smooth,
            "max_torso_lean": lean_smooth,
            "max_knee_diff": None,
            "max_hip_diff": None,
        }

    def _update_rep_metrics(self, features, knee_smooth, hip_smooth, lean_smooth):
        if self.rep_metrics is None:
            return

        if knee_smooth is not None:
            if self.rep_metrics["min_knee"] is None:
                self.rep_metrics["min_knee"] = knee_smooth
            else:
                self.rep_metrics["min_knee"] = min(self.rep_metrics["min_knee"], knee_smooth)

        if hip_smooth is not None:
            if self.rep_metrics["min_hip"] is None:
                self.rep_metrics["min_hip"] = hip_smooth
            else:
                self.rep_metrics["min_hip"] = min(self.rep_metrics["min_hip"], hip_smooth)

        if lean_smooth is not None:
            if self.rep_metrics["max_torso_lean"] is None:
                self.rep_metrics["max_torso_lean"] = lean_smooth
            else:
                self.rep_metrics["max_torso_lean"] = max(self.rep_metrics["max_torso_lean"], lean_smooth)

        knee_diff = features.get("knee_angle_diff")
        hip_diff = features.get("hip_angle_diff")

        if knee_diff is not None:
            if self.rep_metrics["max_knee_diff"] is None:
                self.rep_metrics["max_knee_diff"] = knee_diff
            else:
                self.rep_metrics["max_knee_diff"] = max(self.rep_metrics["max_knee_diff"], knee_diff)

        if hip_diff is not None:
            if self.rep_metrics["max_hip_diff"] is None:
                self.rep_metrics["max_hip_diff"] = hip_diff
            else:
                self.rep_metrics["max_hip_diff"] = max(self.rep_metrics["max_hip_diff"], hip_diff)

    def _finalize_rep(self, knee_smooth, hip_smooth):
        errors = []

        min_knee = self.rep_metrics["min_knee"]
        end_knee = knee_smooth
        end_hip = hip_smooth
        max_knee_diff = self.rep_metrics["max_knee_diff"]
        max_hip_diff = self.rep_metrics["max_hip_diff"]
        max_torso_lean = self.rep_metrics["max_torso_lean"]

        # 1) depth kontrolü
        if min_knee is None or min_knee > self.cfg.half_depth_threshold:
            errors.append("half_depth")

        # 2) lockout kontrolü
        knee_not_locked = end_knee is not None and end_knee < self.cfg.knee_lockout_threshold
        hip_not_locked = end_hip is not None and end_hip < self.cfg.hip_lockout_threshold
        if knee_not_locked or hip_not_locked:
            errors.append("incomplete_lockout")

        # 3) knee_error için ilk basit proxy:
        # sağ-sol belirgin diz / kalça açısı farkı
        if (
            (max_knee_diff is not None and max_knee_diff > self.cfg.knee_diff_threshold) or
            (max_hip_diff is not None and max_hip_diff > self.cfg.hip_diff_threshold)
        ):
            errors.append("knee_error")

        self.rep_count += 1

        feedback = RepFeedback(
            rep_count=self.rep_count,
            has_error=len(errors) > 0,
            error_labels=errors,
            min_knee_angle=min_knee,
            end_knee_angle=end_knee,
            end_hip_angle=end_hip,
            max_knee_angle_diff=max_knee_diff,
            max_hip_angle_diff=max_hip_diff,
            max_torso_lean_deg=max_torso_lean,
        )

        self.rep_metrics = None
        return feedback

    def update(self, result, timestamp_ms=None):
        features = compute_live_frame_features(result, self.cfg)
        if features is None:
            return None

        avg_knee = features["avg_knee_angle"]
        avg_hip = features["avg_hip_angle"]
        avg_lean = features["avg_torso_lean_deg"]

        if avg_knee is None:
            return None

        self.knee_hist.append(avg_knee)
        knee_smooth = safe_mean(list(self.knee_hist))

        if avg_hip is not None:
            self.hip_hist.append(avg_hip)
        hip_smooth = safe_mean(list(self.hip_hist))

        if avg_lean is not None:
            self.lean_hist.append(avg_lean)
        lean_smooth = safe_mean(list(self.lean_hist))

        if self.cooldown > 0:
            self.cooldown -= 1

        # ------------------------------
        # UP -> DOWN
        # ------------------------------
        if self.state == "UP":
            if self.cooldown == 0 and knee_smooth is not None and knee_smooth < self.cfg.descend_threshold:
                self.state = "DOWN"
                self._start_rep_metrics(knee_smooth, hip_smooth, lean_smooth)
                self._update_rep_metrics(features, knee_smooth, hip_smooth, lean_smooth)
            return None

        # ------------------------------
        # DOWN durumunda metrik biriktir
        # ------------------------------
        if self.state == "DOWN":
            self._update_rep_metrics(features, knee_smooth, hip_smooth, lean_smooth)

            # gerçek ROM var mı
            start_knee = self.rep_metrics["start_knee"]
            min_knee = self.rep_metrics["min_knee"]

            rom = None
            if start_knee is not None and min_knee is not None:
                rom = start_knee - min_knee

            # tekrar yukarı çıkıldıysa rep'i kapat
            if knee_smooth is not None and knee_smooth > self.cfg.up_threshold:
                self.state = "UP"
                self.cooldown = self.cfg.cooldown_frames

                # küçük titreşimleri rep saymamak için
                if rom is None or rom < self.cfg.min_rom_deg:
                    self.rep_metrics = None
                    return None

                return self._finalize_rep(knee_smooth, hip_smooth)
    def update_live(self, result, timestamp_ms=None):
        features = compute_live_frame_features(result, self.cfg)
        if features is None:
            return None

        avg_knee = features["avg_knee_angle"]
        avg_hip = features["avg_hip_angle"]
        avg_lean = features["avg_torso_lean_deg"]

        if avg_knee is None:
            return None

        self.knee_hist.append(avg_knee)
        knee_smooth = safe_mean(list(self.knee_hist))

        if avg_hip is not None:
            self.hip_hist.append(avg_hip)
        hip_smooth = safe_mean(list(self.hip_hist))

        if avg_lean is not None:
            self.lean_hist.append(avg_lean)
        lean_smooth = safe_mean(list(self.lean_hist))

        if self.cooldown > 0:
            self.cooldown -= 1

        rep_feedback = None

        # ------------------------------
        # UP -> DOWN
        # ------------------------------
        if self.state == "UP":
            if self.cooldown == 0 and knee_smooth is not None and knee_smooth < self.cfg.descend_threshold:
                self.state = "DOWN"
                self._start_rep_metrics(knee_smooth, hip_smooth, lean_smooth)
                self._update_rep_metrics(features, knee_smooth, hip_smooth, lean_smooth)

        # ------------------------------
        # DOWN durumunda
        # ------------------------------
        elif self.state == "DOWN":
            self._update_rep_metrics(features, knee_smooth, hip_smooth, lean_smooth)

            start_knee = self.rep_metrics["start_knee"]
            min_knee = self.rep_metrics["min_knee"]

            rom = None
            if start_knee is not None and min_knee is not None:
                rom = start_knee - min_knee

            if knee_smooth is not None and knee_smooth > self.cfg.up_threshold:
                self.state = "UP"
                self.cooldown = self.cfg.cooldown_frames

                if rom is None or rom < self.cfg.min_rom_deg:
                    self.rep_metrics = None
                else:
                    rep_feedback = self._finalize_rep(knee_smooth, hip_smooth)

        live_warnings = self._compute_live_warnings(features, knee_smooth, hip_smooth, lean_smooth)

        self.prev_knee_smooth = knee_smooth

        return LiveAnalysis(
            rep_feedback=rep_feedback,
            live_warnings=live_warnings,
            state=self.state,
            avg_knee_angle=knee_smooth,
            avg_hip_angle=hip_smooth,
            avg_torso_lean_deg=lean_smooth,
        )

        return None


_ENGINE = SquatRuleEngine()

def analyze_frame_live(result, timestamp_ms=None):
    return _ENGINE.update_live(result, timestamp_ms=timestamp_ms)

def analyze_frame(result, timestamp_ms=None):
    live = _ENGINE.update_live(result, timestamp_ms=timestamp_ms)
    if live is None:
        return None
    return live.rep_feedback

def reset_analyzer():
    _ENGINE.reset()