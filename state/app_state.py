"""
AppState - Single Source of Truth for CVLA

This is the ONLY place where application state lives.
All UI reads from here. All changes go through the reducer.

INVARIANTS:
1. AppState is immutable (use dataclasses.replace to update)
2. All nested data uses immutable types (tuples, frozen dataclasses)
3. History is tuple-based with max depth
4. No UI component may modify AppState directly
"""

from dataclasses import dataclass, replace, field
from typing import Tuple, Optional, List
import numpy as np

from .models import VectorData, MatrixData, PlaneData, ImageData, EducationalStep


# Maximum undo history depth
MAX_HISTORY = 20


@dataclass(frozen=True)
class AppState:
    """
    Single source of truth for all application state.

    All fields are immutable. To update, use dataclasses.replace().
    """

    # =========================================================================
    # SCENE STATE (domain data)
    # =========================================================================
    vectors: Tuple[VectorData, ...] = ()
    matrices: Tuple[MatrixData, ...] = ()
    planes: Tuple[PlaneData, ...] = ()

    # =========================================================================
    # SELECTION STATE
    # =========================================================================
    selected_id: Optional[str] = None
    selected_type: Optional[str] = None  # 'vector', 'matrix', 'plane', 'image'

    # =========================================================================
    # IMAGE / VISION STATE
    # =========================================================================
    current_image: Optional[ImageData] = None
    processed_image: Optional[ImageData] = None
    selected_kernel: str = 'sobel_x'

    # =========================================================================
    # EDUCATIONAL PIPELINE STATE
    # =========================================================================
    pipeline_steps: Tuple[EducationalStep, ...] = ()
    pipeline_step_index: int = 0

    # =========================================================================
    # UI INPUT STATE (controlled inputs)
    # These mirror what's in the input fields, NOT the saved data.
    # =========================================================================
    input_vector_coords: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    input_vector_label: str = ""
    input_vector_color: Tuple[float, float, float] = (0.8, 0.2, 0.2)

    input_matrix: Tuple[Tuple[float, ...], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    input_matrix_label: str = "A"
    input_matrix_size: int = 3

    input_image_path: str = ""
    input_sample_pattern: str = "checkerboard"
    input_sample_size: int = 32
    input_transform_rotation: float = 0.0
    input_transform_scale: float = 1.0

    # =========================================================================
    # UI VIEW STATE
    # =========================================================================
    active_tab: str = "vectors"
    show_matrix_editor: bool = False
    show_matrix_values: bool = False
    preview_enabled: bool = False

    # =========================================================================
    # HISTORY (for undo/redo)
    # Tuple-based, max depth enforced
    # =========================================================================
    history: Tuple['AppState', ...] = ()
    future: Tuple['AppState', ...] = ()  # For redo

    # =========================================================================
    # COUNTERS (for auto-naming)
    # =========================================================================
    next_vector_id: int = 1
    next_matrix_id: int = 1
    next_color_index: int = 0


def create_initial_state() -> AppState:
    """
    Create the initial application state.

    This is called once at startup.
    """
    # Default basis vectors
    initial_vectors = (
        VectorData.create((2.0, 0.0, 0.0), (1.0, 0.25, 0.25), "i"),
        VectorData.create((0.0, 2.0, 0.0), (0.25, 1.0, 0.25), "j"),
        VectorData.create((0.0, 0.0, 2.0), (0.35, 0.55, 1.0), "k"),
    )

    return AppState(
        vectors=initial_vectors,
        next_vector_id=4,  # Next ID after i, j, k
    )


# =============================================================================
# STATE QUERY HELPERS (Pure functions, no mutation)
# =============================================================================

def get_vector_by_id(state: AppState, id: str) -> Optional[VectorData]:
    """Find a vector by ID."""
    for v in state.vectors:
        if v.id == id:
            return v
    return None


def get_matrix_by_id(state: AppState, id: str) -> Optional[MatrixData]:
    """Find a matrix by ID."""
    for m in state.matrices:
        if m.id == id:
            return m
    return None


def get_selected_vector(state: AppState) -> Optional[VectorData]:
    """Get the currently selected vector, if any."""
    if state.selected_type == 'vector' and state.selected_id:
        return get_vector_by_id(state, state.selected_id)
    return None


def get_selected_matrix(state: AppState) -> Optional[MatrixData]:
    """Get the currently selected matrix, if any."""
    if state.selected_type == 'matrix' and state.selected_id:
        return get_matrix_by_id(state, state.selected_id)
    return None


def get_current_step(state: AppState) -> Optional[EducationalStep]:
    """Get the current educational step."""
    if 0 <= state.pipeline_step_index < len(state.pipeline_steps):
        return state.pipeline_steps[state.pipeline_step_index]
    return None


# =============================================================================
# COLOR PALETTE (for auto-coloring)
# =============================================================================

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


def get_next_color(state: AppState) -> Tuple[Tuple[float, float, float], int]:
    """Get next color from palette and the new index."""
    color = COLOR_PALETTE[state.next_color_index % len(COLOR_PALETTE)]
    new_index = (state.next_color_index + 1) % len(COLOR_PALETTE)
    return color, new_index
