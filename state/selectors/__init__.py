"""
State query helpers and color palette.
"""

from collections import OrderedDict
from math import acos, degrees, sqrt
from typing import Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from state.app_state import AppState
from state.models import VectorData, MatrixData, EducationalStep, TensorData
from state.models.tensor_compat import tensor_to_vector, tensor_to_matrix

# Import tensor selectors
from state.selectors.tensor_selectors import (
    get_tensor_by_id,
    get_selected_tensor,
    get_tensors_by_type,
    get_vectors,
    get_matrices,
    get_images,
    get_visible_tensors,
    get_visible_vectors,
    get_visible_matrices,
    get_visible_images,
    get_tensor_count,
    get_tensor_count_by_type,
    get_tensor_labels,
    get_tensor_ids,
    is_tensor_selected,
    has_tensors,
    has_vectors,
    has_matrices,
    has_images,
    get_tensor_magnitude,
    get_tensor_norm,
    get_tensor_stats,
    get_operation_history,
    get_last_operation,
    get_preview_tensor,
    has_preview,
    get_pending_operation,
    get_active_input_method,
    get_text_input_content,
    get_text_input_type,
    get_grid_size,
    get_grid_cells,
    get_file_path,
    can_create_tensor_from_text,
    can_create_tensor_from_file,
    can_create_tensor_from_grid,
)


def get_vector_by_id(state: "AppState", id: str) -> Optional[VectorData]:
    """Find a vector by ID (checks both legacy vectors and tensors)."""
    # Check legacy vectors first
    for v in state.vectors:
        if v.id == id:
            return v
    # Check tensors (rank-1 = vector)
    for t in state.tensors:
        if t.id == id and t.rank == 1:
            return tensor_to_vector(t)
    return None


def get_matrix_by_id(state: "AppState", id: str) -> Optional[MatrixData]:
    """Find a matrix by ID (checks both legacy matrices and tensors)."""
    # Check legacy matrices first
    for m in state.matrices:
        if m.id == id:
            return m
    # Check tensors (rank-2, non-image = matrix)
    for t in state.tensors:
        if t.id == id and t.rank == 2 and not t.is_image_dtype:
            return tensor_to_matrix(t)
    return None


def get_selected_vector(state: "AppState") -> Optional[VectorData]:
    """Get the currently selected vector, if any (checks both legacy and tensor selection)."""
    # Check legacy selection first
    if state.selected_type == 'vector' and state.selected_id:
        return get_vector_by_id(state, state.selected_id)
    # Check tensor selection (selected_tensor_id pointing to rank-1 tensor)
    if state.selected_tensor_id:
        for t in state.tensors:
            if t.id == state.selected_tensor_id and t.rank == 1:
                return tensor_to_vector(t)
    return None


def get_selected_matrix(state: "AppState") -> Optional[MatrixData]:
    """Get the currently selected matrix, if any (checks both legacy and tensor selection)."""
    # Check legacy selection first
    if state.selected_type == 'matrix' and state.selected_id:
        return get_matrix_by_id(state, state.selected_id)
    # Check tensor selection (selected_tensor_id pointing to rank-2 non-image tensor)
    if state.selected_tensor_id:
        for t in state.tensors:
            if t.id == state.selected_tensor_id and t.rank == 2 and not t.is_image_dtype:
                return tensor_to_matrix(t)
    return None


def get_current_step(state: "AppState") -> Optional[EducationalStep]:
    """Get the current educational step."""
    if 0 <= state.pipeline_step_index < len(state.pipeline_steps):
        return state.pipeline_steps[state.pipeline_step_index]
    return None


_VECTOR_METRIC_CACHE = OrderedDict()
_VECTOR_CACHE_MAX = 256


def _cache_get(cache, key):
    if key in cache:
        value = cache.pop(key)
        cache[key] = value
        return value
    return None


def _cache_set(cache, key, value):
    cache[key] = value
    if len(cache) > _VECTOR_CACHE_MAX:
        cache.popitem(last=False)


def get_vector_magnitude(vector: Union[VectorData, TensorData]) -> float:
    """Get cached vector magnitude (accepts both VectorData and TensorData)."""
    coords = vector.coords if hasattr(vector, 'coords') else vector.data
    key = ("mag", vector.id, coords)
    cached = _cache_get(_VECTOR_METRIC_CACHE, key)
    if cached is not None:
        return cached
    # Pad to 3D if needed
    c = list(coords)
    while len(c) < 3:
        c.append(0.0)
    x, y, z = c[0], c[1], c[2]
    value = sqrt((x * x) + (y * y) + (z * z))
    _cache_set(_VECTOR_METRIC_CACHE, key, value)
    return value


def get_vector_dot_angle(vector_a: Union[VectorData, TensorData], vector_b: Union[VectorData, TensorData]) -> Tuple[float, float]:
    """Get cached dot product and angle between vectors (accepts both VectorData and TensorData)."""
    coords_a = vector_a.coords if hasattr(vector_a, 'coords') else vector_a.data
    coords_b = vector_b.coords if hasattr(vector_b, 'coords') else vector_b.data
    key = ("dot_angle", vector_a.id, coords_a, vector_b.id, coords_b)
    cached = _cache_get(_VECTOR_METRIC_CACHE, key)
    if cached is not None:
        return cached
    if len(coords_a) != len(coords_b):
        value = (0.0, 0.0)
        _cache_set(_VECTOR_METRIC_CACHE, key, value)
        return value
    dot = sum(a * b for a, b in zip(coords_a, coords_b))
    norm_a = sqrt(sum(a * a for a in coords_a))
    norm_b = sqrt(sum(b * b for b in coords_b))
    angle_deg = 0.0
    if norm_a > 1e-10 and norm_b > 1e-10:
        cos_angle = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
        angle_deg = degrees(acos(cos_angle))
    value = (float(dot), float(angle_deg))
    _cache_set(_VECTOR_METRIC_CACHE, key, value)
    return value


def get_vector_axis_projections(vector: Union[VectorData, TensorData]) -> Tuple[float, float, float]:
    """Get cached projections onto axes (accepts both VectorData and TensorData)."""
    vec_coords = vector.coords if hasattr(vector, 'coords') else vector.data
    key = ("proj", vector.id, vec_coords)
    cached = _cache_get(_VECTOR_METRIC_CACHE, key)
    if cached is not None:
        return cached
    coords = list(vec_coords)
    if len(coords) < 3:
        coords += [0.0] * (3 - len(coords))
    value = tuple(float(c) for c in coords[:3])
    _cache_set(_VECTOR_METRIC_CACHE, key, value)
    return value


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
