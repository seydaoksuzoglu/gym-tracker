from dataclasses import dataclass, field
from collections import deque
from typing import List, Optional, Set

from src.analysis.squat_features import extract_squat_features
from src.analysis.squat_counter import SquatCounter
from src.analysis.squat_rules import SquatRules


@dataclass
class RepFeedback:
    rep_count: int
    has_error: bool
    error_labels: List[str] = field(default_factory=list)


@dataclass
class FrameAnalysis:
    state: str
    live_warnings: List[str]
    rep_feedback: Optional[RepFeedback]

    avg_knee_angle: Optional[float]
    avg_hip_angle: Optional[float]
    avg_torso_lean_deg: Optional[float]


class LiveSquatAnalyzer:
    def __init__(self):
        self.counter = SquatCounter()
        self.rules = SquatRules()

        # Rep aktif mi
        self.rep_active: bool = False

        # Rep boyunca biriken hata etiketleri
        self.current_rep_errors: Set[str] = set()

        # Debug için kısa geçmiş
        self.knee_hist = deque(maxlen=5)
        self.hip_hist = deque(maxlen=5)
        self.trunk_hist = deque(maxlen=5)

    def _mean_or_none(self, values):
        if not values:
            return None
        return sum(values) / len(values)

    def analyze(self, result, ts) -> Optional[FrameAnalysis]:
        """
        MediaPipe PoseLandmarkerResult alır.
        Tek kişi üzerinden squat analizi yapar.
        """

        if result is None or not result.pose_landmarks:
            return FrameAnalysis(
                state="no_pose",
                live_warnings=[],
                rep_feedback=None,
                avg_knee_angle=None,
                avg_hip_angle=None,
                avg_torso_lean_deg=None,
            )

        # num_poses=1 olduğu için ilk pozu al
        landmarks = result.pose_landmarks[0]

        # Feature çıkar
        features = extract_squat_features(landmarks)

        print("FEATURE VIEW:", features.view_label)
        print("heel_lift_ratio:", features.heel_lift_ratio)
        print("knee_valgus_offset:", features.knee_valgus_offset)
        print("knee_asymmetry:", features.knee_asymmetry)
        print("hip_below_knee:", features.hip_below_knee)
        print("phase(before update):", self.counter.phase)

        # Düşük güven
        if not features.valid:
            return FrameAnalysis(
                state=f"{self.counter.phase} (low_conf)",
                live_warnings=["Dusuk gorunurluk"],
                rep_feedback=None,
                avg_knee_angle=None,
                avg_hip_angle=None,
                avg_torso_lean_deg=None,
            )

        # Debug history
        self.knee_hist.append(features.knee_angle)
        self.hip_hist.append(features.hip_angle)
        self.trunk_hist.append(features.trunk_angle)

        prev_phase = self.counter.phase

        # Counter güncelle
        event = self.counter.update(features)
        curr_phase = self.counter.phase

        # Canlı kurallar
        live_warnings = self.rules.evaluate(features, self.counter)

        # Rep başlangıcı: standing -> descent
        if prev_phase == "standing" and curr_phase == "descent":
            self.rep_active = True
            self.current_rep_errors.clear()

        # Rep boyunca hataları biriktir
        if self.rep_active and curr_phase in ("descent", "bottom", "ascent"):
            for w in live_warnings:
                self.current_rep_errors.add(w)

        rep_feedback = None

        # Rep tamamlandıysa geri bildirim üret
        if event.rep_completed:
            error_labels = sorted(self.current_rep_errors)

            rep_feedback = RepFeedback(
                rep_count=self.counter.rep_count,
                has_error=len(error_labels) > 0,
                error_labels=error_labels,
            )

            self.current_rep_errors.clear()
            self.rep_active = False

        # Güvenlik amaçlı:
        # Herhangi bir şekilde standing'e dönüldü ama rep tamamlanmadıysa
        # bunu "Tekrar tamamlanamadi" olarak işaretle
        elif self.rep_active and prev_phase in ("descent", "bottom", "ascent") and curr_phase == "standing":
            self.current_rep_errors.add("Tekrar tamamlanamadi")

            rep_feedback = RepFeedback(
                rep_count=self.counter.rep_count + 1,
                has_error=True,
                error_labels=sorted(self.current_rep_errors),
            )

            self.current_rep_errors.clear()
            self.rep_active = False

        return FrameAnalysis(
            state=curr_phase,
            live_warnings=live_warnings,
            rep_feedback=rep_feedback,
            avg_knee_angle=self._mean_or_none(self.knee_hist),
            avg_hip_angle=self._mean_or_none(self.hip_hist),
            avg_torso_lean_deg=self._mean_or_none(self.trunk_hist),
        )