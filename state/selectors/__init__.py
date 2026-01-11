"""
State query helpers and color palette.
"""

from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from state.app_state import AppState
from state.models import VectorData, MatrixData, EducationalStep


def get_vector_by_id(state: "AppState", id: str) -> Optional[VectorData]:
    """Find a vector by ID."""
    for v in state.vectors:
        if v.id == id:
            return v
    return None


def get_matrix_by_id(state: "AppState", id: str) -> Optional[MatrixData]:
    """Find a matrix by ID."""
    for m in state.matrices:
        if m.id == id:
            return m
    return None


def get_selected_vector(state: "AppState") -> Optional[VectorData]:
    """Get the currently selected vector, if any."""
    if state.selected_type == 'vector' and state.selected_id:
        return get_vector_by_id(state, state.selected_id)
    return None


def get_selected_matrix(state: "AppState") -> Optional[MatrixData]:
    """Get the currently selected matrix, if any."""
    if state.selected_type == 'matrix' and state.selected_id:
        return get_matrix_by_id(state, state.selected_id)
    return None


def get_current_step(state: "AppState") -> Optional[EducationalStep]:
    """Get the current educational step."""
    if 0 <= state.pipeline_step_index < len(state.pipeline_steps):
        return state.pipeline_steps[state.pipeline_step_index]
    return None


COLOR_PALETTE = (
    (0.8, 0.2, 0.2),   # Red
    (0.2, 0.8, 0.2),   # Green
    (0.2, 0.2, 0.8),   # Blue
    (0.8, 0.8, 0.2),   # Yellow
    (0.8, 0.2, 0.8),   # Magenta
    (0.2, 0.8, 0.8),   # Cyan
    (0.8, 0.5, 0.2),   # Orange
    (0.5, 0.2, 0.8),   # Purple
)


def get_next_color(state: "AppState") -> Tuple[Tuple[float, float, float], int]:
    """Get next color from palette and the new index."""
    color = COLOR_PALETTE[state.next_color_index % len(COLOR_PALETTE)]
    new_index = (state.next_color_index + 1) % len(COLOR_PALETTE)
    return color, new_index
