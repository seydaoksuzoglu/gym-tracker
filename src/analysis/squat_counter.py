from dataclasses import dataclass
from typing import Optional

from src.analysis.squat_features import SquatFeatures


@dataclass
class SquatEvent:
    rep_completed: bool = False
    phase_changed: bool = False


@dataclass
class SquatCounter:
    stand_knee_angle: float = 155.0
    bottom_knee_angle: float = 110.0

    rep_count: int = 0
    phase: str = "standing"

    prev_knee_angle: Optional[float] = None
    prev_hip_y: Optional[float] = None

    bottom_reached: bool = False
    min_knee_in_rep: float = 180.0

    standing_trunk_baseline: Optional[float] = None

    def _set_phase(self, new_phase):
        changed = self.phase != new_phase
        self.phase = new_phase
        return changed

    def _reset_rep_metrics(self):
        self.bottom_reached = False
        self.min_knee_in_rep = 180.0

    def _update_standing_baseline(self, features: SquatFeatures):
        if self.standing_trunk_baseline is None:
            self.standing_trunk_baseline = features.trunk_angle
        else:
            # hafif yumuşatma
            self.standing_trunk_baseline = (
                0.9 * self.standing_trunk_baseline + 0.1 * features.trunk_angle
            )

    def update(self, features: SquatFeatures) -> SquatEvent:
        event = SquatEvent()

        if not features.valid:
            return event

        if self.phase == "standing":
            self._update_standing_baseline(features)

        knee_falling = (
            self.prev_knee_angle is not None
            and features.knee_angle < self.prev_knee_angle - 2.0
        )
        knee_rising = (
            self.prev_knee_angle is not None
            and features.knee_angle > self.prev_knee_angle + 2.0
        )

        hip_descending = (
            self.prev_hip_y is not None and features.hip_y > self.prev_hip_y + 0.002
        )
        hip_ascending = (
            self.prev_hip_y is not None and features.hip_y < self.prev_hip_y - 0.002
        )

        self.min_knee_in_rep = min(self.min_knee_in_rep, features.knee_angle)

        if self.phase == "standing":
            if knee_falling and hip_descending and features.knee_angle < 150:
                event.phase_changed = self._set_phase("descent")

        elif self.phase == "descent":
            if features.knee_angle <= self.bottom_knee_angle:
                self.bottom_reached = True
                event.phase_changed = self._set_phase("bottom")
            elif knee_rising and self.min_knee_in_rep < 130:
                self.bottom_reached = True
                event.phase_changed = self._set_phase("ascent")

        elif self.phase == "bottom":
            if knee_rising or hip_ascending:
                event.phase_changed = self._set_phase("ascent")

        elif self.phase == "ascent":
            if features.knee_angle > self.stand_knee_angle and not hip_descending:
                event.phase_changed = self._set_phase("standing")

                if self.bottom_reached:
                    self.rep_count += 1
                    event.rep_completed = True

                self._reset_rep_metrics()

        self.prev_knee_angle = features.knee_angle
        self.prev_hip_y = features.hip_y

        return event