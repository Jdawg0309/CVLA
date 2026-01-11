"""History helpers for UI and runtime."""

from state.app_state import AppState


def can_undo(state: AppState) -> bool:
    return bool(state.history)


def can_redo(state: AppState) -> bool:
    return bool(state.future)
