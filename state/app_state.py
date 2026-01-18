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
from state.models.image_model import ImageData
from state.models.educational_step import EducationalStep
from state.models.tensor_model import TensorData
from state.models.operation_record import OperationRecord


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

    # =========================================================================
    # UNIFIED TENSOR STATE (new model - coexists with legacy for migration)
    # =========================================================================
    tensors: Tuple[TensorData, ...] = ()
    selected_tensor_id: Optional[str] = None

    # =========================================================================
    # INPUT PANEL STATE
    # =========================================================================
    active_input_method: str = "text"  # "text", "file", "grid"
    input_text_content: str = ""
    input_text_parsed_type: str = ""  # What the parser detected: "vector", "matrix", ""
    input_file_path: str = ""
    input_grid_rows: int = 3
    input_grid_cols: int = 3
    input_grid_cells: Tuple[Tuple[float, ...], ...] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )

    # =========================================================================
    # OPERATIONS PANEL STATE
    # =========================================================================
    pending_operation: Optional[str] = None
    pending_operation_params: Tuple[Tuple[str, str], ...] = ()
    operation_preview_tensor: Optional[TensorData] = None
    show_operation_preview: bool = True

    # =========================================================================
    # OPERATION HISTORY (for timeline)
    # =========================================================================
    operation_history: Tuple[OperationRecord, ...] = ()

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
    selected_pixel: Optional[Tuple[int, int]] = None
    image_render_mode: str = "plane"  # 'plane' | 'height-field'
    image_render_scale: float = 1.0
    image_color_mode: str = "rgb"  # 'rgb' | 'grayscale' | 'heatmap'
    image_auto_fit: bool = True
    show_image_grid_overlay: bool = False
    image_downsample_enabled: bool = False
    image_preview_resolution: int = 128
    image_max_resolution: int = 512
    current_image_stats: Optional[Tuple[float, float, float, float]] = None
    processed_image_stats: Optional[Tuple[float, float, float, float]] = None
    current_image_preview: Optional[Tuple[Tuple[float, ...], ...]] = None
    processed_image_preview: Optional[Tuple[Tuple[float, ...], ...]] = None
    selected_kernel_matrix: Optional[Tuple[Tuple[float, ...], ...]] = None

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
    input_matrix_rows: int = 3
    input_matrix_cols: int = 3

    input_equations: Tuple[Tuple[float, ...], ...] = (
        (1.0, 1.0, 1.0, 0.0),
        (2.0, -1.0, 0.0, 0.0),
        (0.0, 1.0, -1.0, 0.0),
    )
    input_equation_count: int = 3

    input_image_path: str = ""
    input_sample_pattern: str = "checkerboard"
    input_sample_size: int = 32
    input_transform_rotation: float = 0.0
    input_transform_scale: float = 1.0
    input_image_normalize_mean: float = 0.0
    input_image_normalize_std: float = 1.0
    active_image_tab: str = "raw"
    input_expression: str = ""
    input_expression_type: str = ""
    input_expression_error: str = ""
    input_matrix_preview_vectors: Tuple[Tuple[float, ...], ...] = ()

    # =========================================================================
    # UI VIEW STATE
    # =========================================================================
    active_mode: str = "vectors"
    active_tab: str = "vectors"
    ui_theme: str = "dark"
    active_tool: str = "select"
    show_matrix_editor: bool = False
    show_matrix_values: bool = False
    show_image_on_grid: bool = True
    preview_enabled: bool = False

    matrix_plot_enabled: bool = True
    view_preset: str = "cube"
    view_up_axis: str = "z"
    view_grid_mode: str = "cube"
    view_grid_plane: str = "xy"
    view_show_grid: bool = True
    view_show_axes: bool = True
    view_show_labels: bool = True
    view_grid_size: int = 15
    view_base_major_tick: int = 5
    view_base_minor_tick: int = 1
    view_major_tick: int = 5
    view_minor_tick: int = 1
    view_auto_rotate: bool = False
    view_rotation_speed: float = 0.5
    view_show_cube_faces: bool = True
    view_show_cube_corners: bool = True
    view_cubic_grid_density: float = 1.0
    view_cube_face_opacity: float = 0.05
    view_mode_2d: bool = False

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
    selected_kernel_matrix = None
    try:
        from domain.images import get_kernel_by_name
        kernel = get_kernel_by_name("sobel_x")
        selected_kernel_matrix = tuple(tuple(float(v) for v in row) for row in kernel)
    except Exception:
        selected_kernel_matrix = None

    return AppState(
        vectors=(),
        next_vector_id=1,
        selected_kernel_matrix=selected_kernel_matrix,
    )
