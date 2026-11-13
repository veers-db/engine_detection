from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .context import FSMContext


class State(ABC):
    """Abstract base class for FSM states."""

    def __init__(self, name: str) -> None:
        self.name = name

    def on_enter(self, context: FSMContext) -> None:
        pass

    @abstractmethod
    def execute(self, context: FSMContext) -> Optional[State]:
        """Returns the next State to transition to, or None to stay in the current state."""

    def on_exit(self, context: FSMContext) -> None:
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.name})>"
