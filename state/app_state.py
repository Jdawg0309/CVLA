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

from state.models.vector_model import VectorData
from state.models.matrix_model import MatrixData
from state.models.plane_model import PlaneData
from state.models.image_model import ImageData
from state.models.educational_step import EducationalStep
from state.models.pipeline_models import PipelineOp, MicroOp


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
    image_status: str = ""
    image_status_level: str = "info"  # "info" | "error"
    image_pipeline: Tuple[PipelineOp, ...] = ()
    active_pipeline_index: int = 0
    micro_step_index: int = 0
    micro_step_total: int = 0
    micro_op: Optional[MicroOp] = None
    selected_pixel: Optional[Tuple[int, int]] = None
    image_step_index: int = 0
    image_step_total: int = 0
    image_render_mode: str = "plane"  # 'plane' | 'height-field'
    image_render_scale: float = 1.0
    image_color_mode: str = "rgb"  # 'rgb' | 'grayscale' | 'heatmap'
    image_auto_fit: bool = True
    show_image_grid_overlay: bool = False
    image_downsample_enabled: bool = False
    image_preview_resolution: int = 128
    image_max_resolution: int = 512

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
    input_mnist_index: int = 0
    input_sample_pattern: str = "checkerboard"
    input_sample_size: int = 32
    input_transform_rotation: float = 0.0
    input_transform_scale: float = 1.0
    input_image_normalize_mean: float = 0.0
    input_image_normalize_std: float = 1.0
    active_image_tab: str = "raw"

    # =========================================================================
    # UI VIEW STATE
    # =========================================================================
    active_tab: str = "vectors"
    ribbon_tab: str = "File"
    ui_theme: str = "dark"
    active_tool: str = "select"
    show_matrix_editor: bool = False
    show_matrix_values: bool = False
    show_heatmap: bool = True
    show_channels: bool = False
    show_image_on_grid: bool = True
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

    This is called once at startup. This is the SINGLE SOURCE OF TRUTH
    for default vectors - do not duplicate elsewhere.
    """
    # Default basis vectors + example vectors
    initial_vectors = (
        VectorData.create((2.0, 0.0, 0.0), (1.0, 0.25, 0.25), "i"),
        VectorData.create((0.0, 2.0, 0.0), (0.25, 1.0, 0.25), "j"),
        VectorData.create((0.0, 0.0, 2.0), (0.35, 0.55, 1.0), "k"),
        VectorData.create((1.0, 1.0, 0.0), (0.8, 0.6, 0.2), "v1"),
        VectorData.create((0.5, 1.0, 1.0), (0.6, 0.2, 0.8), "v2"),
    )

    return AppState(
        vectors=initial_vectors,
        next_vector_id=6,  # Next ID after i, j, k, v1, v2
    )
