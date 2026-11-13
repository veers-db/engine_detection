from __future__ import annotations

from typing import Optional

from framework.fsm.context import FSMContext
from framework.fsm.state import State


class EnginePresentState(State):
    """
    Engine is confirmed present at the station.
    Absorbs momentary occlusions (< `debounce_threshold` frames) before transitioning back to absent.
    """

    def __init__(self) -> None:
        super().__init__(name="ENGINE_PRESENT")

    def on_enter(self, context: FSMContext) -> None:
        context.current_status = self.name
        context.consecutive_missing = 0

    def execute(self, context: FSMContext) -> Optional[State]:
        from .engine_absent_state import EngineAbsentState

        if context.has_engine:
            context.consecutive_detections += 1
            context.consecutive_missing = 0
        else:
            context.consecutive_missing += 1
            context.consecutive_detections = 0
            if context.consecutive_missing >= context.debounce_threshold:
                return EngineAbsentState()

        return None
