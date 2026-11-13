from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List


@dataclass(frozen=True)
class StateTransitionEvent:
    from_state: str
    to_state: str
    frame_idx: int
    timestamp: float = field(default_factory=time.time)
    reason: str = ""


@dataclass
class FSMContext:
    """Holds frame-level detection telemetry, debounce counters, and transition history."""

    debounce_threshold: int = 10
    current_frame_idx: int = 0
    has_engine: bool = False
    raw_detections: Any = None
    consecutive_detections: int = 0
    consecutive_missing: int = 0
    current_status: str = "ENGINE_ABSENT"
    transition_history: List[StateTransitionEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.debounce_threshold < 1:
            raise ValueError(f"debounce_threshold must be >= 1, got {self.debounce_threshold}")

    def update_frame(self, frame_idx: int, has_engine: bool, raw_detections: Any = None) -> None:
        self.current_frame_idx = frame_idx
        self.has_engine = has_engine
        self.raw_detections = raw_detections

    def record_transition(self, from_state: str, to_state: str, reason: str = "") -> None:
        self.transition_history.append(
            StateTransitionEvent(
                from_state=from_state,
                to_state=to_state,
                frame_idx=self.current_frame_idx,
                reason=reason,
            )
        )
