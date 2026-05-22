"""
Faz dizisini takip eder, tam turu rep'e cevirir.
Cycle kapanis tetigi: DESCENT -> {SETUP, PULL} gecisi.
PhaseEvent.from_phase kullanarak temiz gecis bilgisi alir.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from src.analysis.deadlift.phase_detector import Phase, PhaseEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepEvent:
    rep_completed: bool = False
    rep_id: int = 0
    incomplete: bool = False
    incomplete_reason: str = ""
    phase_durations_ms: Optional[Dict[str, int]] = None


@dataclass
class RepCounter:
    rep_count: int = 0
    incomplete_count: int = 0

    _phase_durations: Dict[Phase, int] = field(default_factory=dict)
    _visited_in_cycle: Set[Phase] = field(default_factory=set)
    _prev_ts: Optional[int] = None
    _current_phase: Optional[Phase] = None

    def update(self, event: PhaseEvent) -> RepEvent:
        # Faz suresi akumulasyonu (her frame, gecis olmasa da)
        if self._current_phase is not None and self._prev_ts is not None:
            delta = event.ts_ms - self._prev_ts
            if delta > 0:
                self._phase_durations[self._current_phase] = (
                    self._phase_durations.get(self._current_phase, 0) + delta
                )
        self._current_phase = event.phase
        self._prev_ts = event.ts_ms

        if not event.phase_changed:
            return RepEvent()

        # from_phase PhaseEvent'te commit oncesi yakalanmis durumda
        if event.from_phase is not None:
            self._visited_in_cycle.add(event.from_phase)
        self._visited_in_cycle.add(event.phase)

        # Cycle kapanis tetigi: DESCENT -> (SETUP veya PULL)
        is_cycle_close = (
            event.from_phase == Phase.DESCENT
            and event.phase in (Phase.SETUP, Phase.PULL)
        )
        if is_cycle_close:
            return self._close_cycle()

        return RepEvent()

    def _close_cycle(self) -> RepEvent:
        visited = self._visited_in_cycle
        had_pull = Phase.PULL in visited
        had_lockout = Phase.LOCKOUT in visited
        had_descent = Phase.DESCENT in visited

        durations = {p.value: self._phase_durations.get(p, 0) for p in Phase}
        self._reset_cycle()

        if had_pull and had_lockout and had_descent:
            self.rep_count += 1
            logger.info("Rep tamamlandi: id=%d", self.rep_count)
            return RepEvent(
                rep_completed=True,
                rep_id=self.rep_count,
                phase_durations_ms=durations,
            )

        self.incomplete_count += 1
        reason = "no_lockout" if not had_lockout else "missing_phase"
        logger.info("Incomplete rep: reason=%s", reason)
        return RepEvent(
            rep_completed=False,
            incomplete=True,
            incomplete_reason=reason,
            phase_durations_ms=durations,
        )

    def _reset_cycle(self) -> None:
        self._phase_durations.clear()
        self._visited_in_cycle.clear()