from dataclasses import dataclass
from collections import defaultdict
from typing import List, Optional

from src.analysis.squat_features import SquatFeatures
from src.analysis.squat_counter import SquatCounter


@dataclass
class RuleThreshold:
    # Genel
    min_visibility: float = 0.55
    min_torso_height: float = 0.07

    # View tahmini
    # frontal_width / torso_height oranı küçükse kişi yanda kabul edilir
    side_view_ratio_threshold: float = 0.25

    # --------------------------------
    # Yandan Görünüm Kuralları
    # --------------------------------

    # Topuk kalkma
    # Not: burada mutlak heel_lift_ratio değil
    # standing baseline'a göre delta kullanıyoruz.
    heel_lift_error_delta: float = 0.15
    heel_lift_persist_frames: int = 6

    # Derinlik
    partial_depth_knee_angle_deg: float = 125.0
    partial_depth_persist_frames: int = 3


    # --------------------------------
    # Önden Görünüm Kuralları
    # --------------------------------

    # Tek başına çok güçlü valgus
    knee_valgus_error_offset: float = 0.70

    # Tek başına çok güçlü asimetri
    knee_asymmetry_error: float = 0.16

    # İkisi birlikte orta seviyede yüksekse de hata ver
    knee_alignment_combo_valgus: float = 0.55
    knee_alignment_combo_asym: float = 0.10

    # Kaç frame üst üste sürerse hata sayılacak
    knee_alignment_persist_frames: int = 6


class SquatRules:
    def __init__(self, thresholds: Optional[RuleThreshold] = None):
        self.t = thresholds or RuleThreshold()

        # Her kural için kaç frame üst üste aktif olduğunu tutar
        self._active_counts = defaultdict(int)

        # Standing anındaki topuk baseline'ı
        self.standing_heel_baseline = None

        # Son view
        self.prev_view = None

    # -------------------------------------------------
    # Yardımcılar
    # -------------------------------------------------

    def _hold(self, rule_name: str, condition: bool) -> int:
        if condition:
            self._active_counts[rule_name] += 1
        else:
            self._active_counts[rule_name] = 0
        return self._active_counts[rule_name]

    def _reset_keys(self, keys: List[str]):
        for k in keys:
            self._active_counts[k] = 0

    def _deduplicate(self, items: List[str]) -> List[str]:
        return list(dict.fromkeys(items))

    def _reset_all_rule_counters(self):
        for k in list(self._active_counts.keys()):
            self._active_counts[k] = 0

    def _update_standing_baselines(self, features: SquatFeatures):
        if self.standing_heel_baseline is None:
            self.standing_heel_baseline = features.heel_lift_ratio
        else:
            self.standing_heel_baseline = (
                0.90 * self.standing_heel_baseline
                + 0.10 * features.heel_lift_ratio
            )

    def _infer_view(self, features: SquatFeatures) -> str:
        """
        Öncelik:
        1) features.view_label varsa onu kullan
        2) yoksa shoulder_span / hip_span / torso_height ile side-front tahmini yap
        """
        explicit_view = getattr(features, "view_label", None)
        if explicit_view in ("side", "front"):
            return explicit_view

        if features.torso_height <= self.t.min_torso_height:
            return "front"

        frontal_width = (features.shoulder_span + features.hip_span) / 2.0
        view_ratio = frontal_width / max(features.torso_height, 1e-6)

        if view_ratio < self.t.side_view_ratio_threshold:
            return "side"
        return "front"

    def _reset_non_relevant_rules(self, phase: str):
        if phase == "standing":
            self._reset_keys([
                "heel_rise_error",
                "partial_depth",
                "knee_alignment_error",
            ])

    # -------------------------------------------------
    # YANDAN KURALLAR
    # -------------------------------------------------

    def _check_heel_rise(self, features: SquatFeatures) -> List[str]:
        warnings = []

        if self.standing_heel_baseline is None:
            return warnings

        heel_delta = max(0.0, features.heel_lift_ratio - self.standing_heel_baseline)
        error_active = heel_delta >= self.t.heel_lift_error_delta

        error_frames = self._hold("heel_rise_error", error_active)

        # Debug için aç
        print("HEEL baseline:", self.standing_heel_baseline)
        print("HEEL ratio:", features.heel_lift_ratio)
        print("HEEL delta:", heel_delta)
        print("HEEL frames:", error_frames)

        if error_frames >= self.t.heel_lift_persist_frames:
            warnings.append("Topuk yerden kalkti")

        return warnings

    def _check_depth(self, features: SquatFeatures) -> List[str]:
        warnings = []

        depth_bad = (
            (not features.hip_below_knee)
            and (features.knee_angle >= self.t.partial_depth_knee_angle_deg)
        )

        active_frames = self._hold("partial_depth", depth_bad)

        if active_frames >= self.t.partial_depth_persist_frames:
            warnings.append("Yeterli derinlige inilmedi")

        return warnings

    # -------------------------------------------------
    # ÖNDEN KURALLAR
    # -------------------------------------------------

    def _check_knee_alignment(self, features: SquatFeatures) -> bool:
        kv = getattr(features, "knee_valgus_offset", 0.0)
        ka = getattr(features, "knee_asymmetry", 0.0)

        strong_valgus = kv >= self.t.knee_valgus_error_offset
        strong_asym = ka >= self.t.knee_asymmetry_error
        combo_bad = (
            kv >= self.t.knee_alignment_combo_valgus
            and ka >= self.t.knee_alignment_combo_asym
        )

        knee_bad = strong_valgus or strong_asym or combo_bad
        frames = self._hold("knee_alignment_error", knee_bad)

        print("KV:", kv, "KA:", ka, "knee_bad:", knee_bad, "frames:", frames)

        return frames >= self.t.knee_alignment_persist_frames

    # -------------------------------------------------
    # ANA evaluate
    # -------------------------------------------------

    def evaluate(self, features: SquatFeatures, counter: SquatCounter) -> List[str]:
        warnings = []

        if (not features.valid) or (features.avg_visibility < self.t.min_visibility):
            return warnings

        if features.torso_height < self.t.min_torso_height:
            return warnings

        # DÜZELTME: string değil, fonksiyon çağrısı olmalı
        view = self._infer_view(features)

        # View değiştiyse sayaçları sıfırla
        if self.prev_view is not None and self.prev_view != view:
            self._reset_all_rule_counters()
        self.prev_view = view

        # Standing baseline güncelle
        if counter.phase == "standing" and features.knee_angle > 160:
            self._update_standing_baselines(features)

        self._reset_non_relevant_rules(counter.phase)

        print("RULE VIEW:", view)
        print("RULE PHASE:", counter.phase)

        # -----------------------------
        # YANDAN KURALLAR
        # -----------------------------
        if view == "side":
            # Topuk ve derinlik esas olarak dipte anlamlı
            if counter.phase == "bottom":
                warnings.extend(self._check_heel_rise(features))
                warnings.extend(self._check_depth(features))
            else:
                self._hold("heel_rise_error", False)
                self._hold("partia_depth", False)

        # -----------------------------
        # ÖNDEN KURALLAR
        # -----------------------------
        elif view == "front":
            if counter.phase == "bottom":
                if self._check_knee_alignment(features):
                    warnings.append("Diz hizasi bozuldu")
            else:
                self._hold("knee_alignment_error", False)

        warnings = self._deduplicate(warnings)
        print("RULE WARNINGS FINAL:", warnings)
        return warnings

        
    
       










