"""
Deadlift pipeline wrapper - run_pose.py icin tek giris noktasi.
Sprint 2: capture + filter + calibrate + phase + rep_counter.
Sprint 3: ckpt sampling + Z-skor (her rep sonu z_scores listesi).
Sprint 4'te error_detectors + scoring eklenir.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.analysis.deadlift.error_detectors.base import ErrorDetector, ErrorReport, RepSummary
from src.analysis.deadlift.error_detectors.incomplete_lockout import IncompleteLockoutDetector
from src.analysis.deadlift.error_detectors.uncontrolled_descent import UncontrolledDescentDetector
from src.analysis.deadlift.scoring import overall_grade

from src.analysis.deadlift.capture import PoseCapture
from src.analysis.deadlift.filters import EMAFilter
from src.analysis.deadlift.calibration import StandingCalibrator
from src.analysis.deadlift.pre_checks import check_side_view
from src.analysis.deadlift.deadlift_features import extract_deadlift_features
from src.analysis.deadlift.phase_detector import PhaseDetector
from src.analysis.deadlift.rep_counter import RepCounter, RepEvent
from src.analysis.deadlift.checkpoints import CheckpointCollector
from src.analysis.deadlift.z_score import (
    CheckpointReference,
    ZScores,
    compute_z,
    load_reference,
)
from src.analysis.deadlift.landmarks import LandmarkFrame

@dataclass
class DeadliftAnalysis:
    """run_pose.py'nin tukettigi cikti."""
    phase: str                              # "setup" | "pull" | "lockout" | "descent" | "calibrating"
    rep_count: int
    incomplete_count: int
    rejected_reason: Optional[str]
    calibrating: bool
    T: Optional[float]
    rep_event: Optional[RepEvent]
    landmarks: Optional[LandmarkFrame]
    hip_dy: Optional[float]
    hip_velocity: Optional[float]
    candidate_phase: Optional[str]
    candidate_held_ms: Optional[int]
    rep_z_scores: Optional[List[ZScores]]   # sadece rep_event.rep_completed/incomplete iken dolu
    live_checkpoint_count: int              # devam eden rep'te su ana kadar yakalanan ckpt sayisi
    rep_errors: Optional[Dict[str, ErrorReport]]   # sadece rep kapaninda dolu
    rep_overall_grade: Optional[str]               # "perfect", "good", "needs_attention", ...

class LiveDeadliftAnalyzer:
    def __init__(self, model_path: str):
        self.capture = PoseCapture(str(model_path), running_mode="VIDEO")
        self.ema = EMAFilter(alpha=0.3)
        self.calibrator = StandingCalibrator(frames_required=30)
        self.phase_detector: Optional[PhaseDetector] = None
        self.rep_counter = RepCounter()
        self.checkpoint_collector = CheckpointCollector()
        self._references: dict[str, CheckpointReference] = load_reference()
        self._current_rep_z_scores: List[ZScores] = []
        self.detectors: List[ErrorDetector] = [
        IncompleteLockoutDetector(),
        UncontrolledDescentDetector(),
        ]
        self._prev_frame_ts: Optional[int] = None

    def analyze(self, frame_rgb, ts_ms: int) -> DeadliftAnalysis:
        raw = self.capture.process(frame_rgb, ts_ms)
        filtered = self.ema.update(raw)

        # === Sprint 4: frame-level dt (kalibrasyon/rejected dahil hep guncellenir) ===
        dt_ms = 0 if self._prev_frame_ts is None else max(0, ts_ms - self._prev_frame_ts)
        self._prev_frame_ts = ts_ms

        # 1) Kalibrasyon
        if not self.calibrator.is_ready():
            self.calibrator.update(filtered)
            return self._calibrating_output(filtered)

        # 2) PhaseDetector lazy init
        if self.phase_detector is None:
            self.phase_detector = PhaseDetector(baseline=self.calibrator.get_baseline())

        # 3) Side-view check
        rejected = check_side_view(filtered, self.calibrator.get_baseline())
        if rejected is not None:
            return self._rejected_output(filtered, rejected.reason)

        # 4) Pipeline: features -> phase -> rep   [DEGISMEZ]
        features = extract_deadlift_features(filtered)
        phase_ev = self.phase_detector.update(features, ts_ms)
        rep_ev = self.rep_counter.update(phase_ev)

        # 4a) Ckpt sampling + Z hesabi   [DEGISMEZ — Sprint 3]
        emitted = self.checkpoint_collector.update(phase_ev.phase, features, ts_ms)
        for sample in emitted:
            ref = self._references.get(sample.id)
            if ref is None:
                continue
            self._current_rep_z_scores.append(compute_z(sample, ref))

        # === 4b) Sprint 4: detektor frame update (YENI EKLEME) ===
        for detector in self.detectors:
            detector.update(features, phase_ev.phase, ts_ms, dt_ms)

        # 4c) Rep kapandiysa buffer'lari disari ver
        #     [MEVCUT 4b genisletildi: + errors + reset]
        rep_z_scores: Optional[List[ZScores]] = None
        rep_errors: Optional[Dict[str, ErrorReport]] = None
        rep_overall_grade: Optional[str] = None

        if rep_ev.rep_completed or rep_ev.incomplete:
            # Sprint 3: ckpt buffer flush
            rep_z_scores = list(self._current_rep_z_scores)
            self._current_rep_z_scores.clear()
            self.checkpoint_collector.start_new_rep()

            # === Sprint 4: detektor evaluation + reset (YENI EKLEME) ===
            summary = RepSummary(
                rep_id=self.rep_counter.rep_count,
                phase_durations_ms=rep_ev.phase_durations_ms or {},
            )
            rep_errors = {}
            for detector in self.detectors:
                report = detector.evaluate_rep(summary)
                if report is not None:
                    rep_errors[report.error_id] = report
                detector.reset()
            rep_overall_grade = overall_grade([r.area for r in rep_errors.values()])

        # 5) Diagnostics + output (mevcut + 2 yeni alan)
        diag = self.phase_detector.get_diagnostics(ts_ms)
        hip_dy = (
            features.hip_y - self.calibrator.get_baseline().standing_hip_y
            if features.valid else None
        )

        return DeadliftAnalysis(
            phase=phase_ev.phase.value,
            rep_count=self.rep_counter.rep_count,
            incomplete_count=self.rep_counter.incomplete_count,
            rejected_reason=None,
            calibrating=False,
            T=features.T if features.valid else None,
            rep_event=rep_ev if (rep_ev.rep_completed or rep_ev.incomplete) else None,
            landmarks=filtered,
            hip_dy=hip_dy,
            hip_velocity=phase_ev.hip_velocity,
            candidate_phase=diag["candidate_phase"],
            candidate_held_ms=diag["candidate_held_ms"],
            rep_z_scores=rep_z_scores,
            live_checkpoint_count=len(self._current_rep_z_scores),
            # === Sprint 4: iki yeni alan ===
            rep_errors=rep_errors,
            rep_overall_grade=rep_overall_grade,
        )

    def close(self):
        self.capture.close()

    # ---- yardimcilar ----

    def _calibrating_output(self, filtered: LandmarkFrame) -> DeadliftAnalysis:
        return DeadliftAnalysis(
            phase="calibrating", rep_count=0, incomplete_count=0,
            rejected_reason=None, calibrating=True,
            T=None, rep_event=None, landmarks=filtered,
            hip_dy=None, hip_velocity=None,
            candidate_phase=None, candidate_held_ms=None,
            rep_z_scores=None, live_checkpoint_count=0,
            rep_errors=None, rep_overall_grade=None,
        )

    def _rejected_output(
        self, filtered: LandmarkFrame, reason: str
    ) -> DeadliftAnalysis:
        return DeadliftAnalysis(
            phase=self.phase_detector.phase.value,
            rep_count=self.rep_counter.rep_count,
            incomplete_count=self.rep_counter.incomplete_count,
            rejected_reason=reason, calibrating=False,
            T=None, rep_event=None, landmarks=filtered,
            hip_dy=None, hip_velocity=None,
            candidate_phase=None, candidate_held_ms=None,
            rep_z_scores=None,
            live_checkpoint_count=len(self._current_rep_z_scores),
            rep_errors=None, rep_overall_grade=None,
        )
