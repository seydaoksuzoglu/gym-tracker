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

        self.standing_kv_baseline = None
        self.standing_ka_baseline = None
        self.bottom_kv_values = []
        self.bottom_ka_values = []

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
        if features.view_label == "front" and self.counter.phase == "standing":
            if self.standing_kv_baseline is None:
                self.standing_kv_baseline = features.knee_valgus_offset
                self.standing_ka_baseline = features.knee_asymmetry
            else:
                self.standing_kv_baseline = (
                    0.9 * self.standing_kv_baseline + 0.1 * features.knee_valgus_offset
                )
                self.standing_ka_baseline = (
                    0.9 * self.standing_ka_baseline + 0.1 * features.knee_asymmetry
                )

        print("FEATURE VIEW:", features.view_label)
        #print("heel_lift_ratio:", features.heel_lift_ratio)
        print("knee_valgus_offset:", features.knee_valgus_offset)
        #print("knee_asymmetry:", features.knee_asymmetry)
        #print("hip_below_knee:", features.hip_below_knee)
        print("phase(before update):", self.counter.phase)

        # Debug history
        self.knee_hist.append(features.knee_angle)
        self.hip_hist.append(features.hip_angle)
        self.trunk_hist.append(features.trunk_angle)

        prev_phase = self.counter.phase

        # Counter güncelle
        event = self.counter.update(features)
        curr_phase = self.counter.phase

        print("phase(after update):", curr_phase)
        print("rep_completed:", event.rep_completed)
        # Canlı kurallar
        live_warnings = self.rules.evaluate(features, self.counter)
        if self.rep_active and curr_phase == "bottom" and features.view_label == "front":
            kv_delta = max(0.0, features.knee_valgus_offset - (self.standing_kv_baseline or 0.0))
            ka_delta = max(0.0, features.knee_asymmetry - (self.standing_ka_baseline or 0.0))

            self.bottom_kv_values.append(kv_delta)
            self.bottom_ka_values.append(ka_delta)

            print("BOTTOM DELTA -> kv:", kv_delta, "ka:", ka_delta)

        # Rep başlangıcı: standing -> descent
        if prev_phase == "standing" and curr_phase == "descent":
            print("NEW REP STARTED -> clearing errors | rep_count:", self.counter.rep_count)
            self.rep_active = True
            self.current_rep_errors.clear()
            self.bottom_kv_values.clear()
            self.bottom_ka_values.clear()

        # Rep boyunca hataları biriktir
        if self.rep_active and curr_phase in ("descent", "bottom", "ascent"):
            for w in live_warnings:
                print("ADD REP ERROR:", w, "phase:", curr_phase)
                self.current_rep_errors.add(w)
        print("CURRENT REP ERRORS:", sorted(self.current_rep_errors))

        rep_feedback = None

        # Rep tamamlandıysa geri bildirim üret
        if event.rep_completed:
            if self.bottom_kv_values:
                kv_sorted = sorted(self.bottom_kv_values)
                kv_max = max(kv_sorted)
                kv_med = kv_sorted[len(kv_sorted) // 2]

                ka_sorted = sorted(self.bottom_ka_values) if self.bottom_ka_values else []
                ka_max = max(ka_sorted) if ka_sorted else 0.0
                ka_med = ka_sorted[len(ka_sorted) // 2] if ka_sorted else 0.0


                print(
                    "REP KNEE SUMMARY ->",
                    "kv_max:", kv_max,
                    "kv_med:", kv_med,
                    "ka_max:", ka_max,
                    "ka_med:", ka_med,
                )

                # Daha guvenli karar
                strong_valgus = (kv_med >= 0.53 and kv_max >= 0.70)
                sustained_combo = (kv_med >= 0.48 and ka_max >= 0.12 and ka_med >= 0.03)

                if strong_valgus or sustained_combo:
                    self.current_rep_errors.add("Diz hizasi bozuldu")

            error_labels = sorted(self.current_rep_errors)
            print("REP COMPLETED, ERRORS:", error_labels)

            rep_feedback = RepFeedback(
                rep_count=self.counter.rep_count,
                has_error=len(error_labels) > 0,
                error_labels=error_labels,
            )

            self.current_rep_errors.clear()
            self.bottom_kv_values.clear()
            self.bottom_ka_values.clear()
            self.rep_active = False

        # Güvenlik amaçlı:
        # Herhangi bir şekilde standing'e dönüldü ama rep tamamlanmadıysa
        # bunu "Tekrar tamamlanamadi" olarak işaretle
        elif self.rep_active and prev_phase in ("descent", "bottom", "ascent") and curr_phase == "standing":
            print("REP FAILED -> Tekrar tamamlanamadi")
            self.current_rep_errors.add("Tekrar tamamlanamadi")
            self.bottom_kv_values.clear()
            self.bottom_ka_values.clear()

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