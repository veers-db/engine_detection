from __future__ import annotations

from typing import Optional

from framework.fsm.context import FSMContext
from framework.fsm.state import State


class EngineAbsentState(State):
    """
    Station is empty or engine is absent.
    Requires `debounce_threshold` consecutive frames of valid detection to transition.
    """

    def __init__(self) -> None:
        super().__init__(name="ENGINE_ABSENT")

    def on_enter(self, context: FSMContext) -> None:
        context.current_status = self.name
        context.consecutive_detections = 0

    def execute(self, context: FSMContext) -> Optional[State]:
        from .engine_present_state import EnginePresentState

        if context.has_engine:
            context.consecutive_detections += 1
            context.consecutive_missing = 0
            if context.consecutive_detections >= context.debounce_threshold:
                return EnginePresentState()
        else:
            context.consecutive_detections = 0
            context.consecutive_missing += 1

        return None
