"""
State store and dispatch helpers.
"""

from typing import Callable

from state.app_state import AppState
from state.actions import Action
from state.reducers import reduce


class Store:
    """
    Simple store that holds state and processes actions.
    """

    def __init__(self, initial_state: AppState):
        self._state = initial_state
        self._listeners: list = []

    def get_state(self) -> AppState:
        """Get current state (read-only)."""
        return self._state

    def dispatch(self, action: Action) -> None:
        """Dispatch an action to update state."""
        self._state = reduce(self._state, action)
        for listener in self._listeners:
            listener(self._state)

    def subscribe(self, listener: Callable[[AppState], None]) -> Callable[[], None]:
        """Subscribe to state changes. Returns unsubscribe function."""
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)
