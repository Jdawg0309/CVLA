"""
Reducer - The ONLY place where state changes happen.
"""

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runtime.app_state import AppState
from state.actions import Action
from state.reducer_history import reduce_history, should_record_history
from state.reducer_vectors import reduce_vectors
from state.reducer_matrices import reduce_matrices
from state.reducer_images import reduce_images
from state.reducer_pipeline import reduce_pipeline
from state.reducer_inputs import reduce_inputs
from state.reducer_navigation import reduce_navigation


def reduce(state: "AppState", action: Action) -> "AppState":
    """
    Pure reducer function.
    """
    history_state = reduce_history(state, action)
    if history_state is not None:
        return history_state

    def with_history(new_state: "AppState") -> "AppState":
        from runtime import app_state as _app_state
        if not should_record_history(action):
            return new_state
        new_history = (state.history + (state,))[-_app_state.MAX_HISTORY:]
        return replace(new_state, history=new_history, future=())

    result = reduce_vectors(state, action, with_history)
    if result is not None:
        return result

    result = reduce_matrices(state, action, with_history)
    if result is not None:
        return result

    result = reduce_images(state, action, with_history)
    if result is not None:
        return result

    result = reduce_pipeline(state, action, with_history)
    if result is not None:
        return result

    result = reduce_inputs(state, action)
    if result is not None:
        return result

    result = reduce_navigation(state, action)
    if result is not None:
        return result

    return state
