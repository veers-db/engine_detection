from __future__ import annotations

import logging
from typing import Optional

from .context import FSMContext
from .state import State

logger = logging.getLogger("EngineFSM")


class StateMachine:
    """Coordinates frame-by-frame state transitions and debounce telemetry."""

    def __init__(self, initial_state: Optional[State] = None, debounce_threshold: int = 10) -> None:
        self.context = FSMContext(debounce_threshold=debounce_threshold)
        self.current_state: Optional[State] = None
        if initial_state is not None:
            self.initialize(initial_state)

    def initialize(self, initial_state: State) -> None:
        self.current_state = initial_state
        self.current_state.on_enter(self.context)

    def step(self, has_engine: bool, raw_detections: Optional[object] = None) -> State:
        if self.current_state is None:
            raise RuntimeError("StateMachine must be initialized before calling step().")

        self.context.update_frame(
            frame_idx=self.context.current_frame_idx + 1,
            has_engine=has_engine,
            raw_detections=raw_detections,
        )

        next_state = self.current_state.execute(self.context)
        if next_state is not None and next_state.name != self.current_state.name:
            self._transition(next_state)

        return self.current_state

    def _transition(self, next_state: State) -> None:
        assert self.current_state is not None
        prev_name, new_name = self.current_state.name, next_state.name

        logger.info(
            f"[FSM TRANSITION] Frame {self.context.current_frame_idx}: {prev_name} -> {new_name} "
            f"(detections: {self.context.consecutive_detections}, missing: {self.context.consecutive_missing})"
        )

        self.current_state.on_exit(self.context)
        self.context.record_transition(
            from_state=prev_name,
            to_state=new_name,
            reason=f"Debounce threshold of {self.context.debounce_threshold} consecutive frames reached",
        )
        self.current_state = next_state
        self.current_state.on_enter(self.context)

    @property
    def status(self) -> str:
        return self.context.current_status

    @property
    def is_engine_present(self) -> bool:
        return self.current_state is not None and self.current_state.name == "ENGINE_PRESENT"
