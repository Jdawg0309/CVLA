"""
State query helpers and color palette.
"""

from collections import OrderedDict
from math import acos, degrees, sqrt
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


def get_vector_magnitude(vector: VectorData) -> float:
    """Get cached vector magnitude."""
    key = ("mag", vector.id, vector.coords)
    cached = _cache_get(_VECTOR_METRIC_CACHE, key)
    if cached is not None:
        return cached
    x, y, z = vector.coords
    value = sqrt((x * x) + (y * y) + (z * z))
    _cache_set(_VECTOR_METRIC_CACHE, key, value)
    return value


def get_vector_dot_angle(vector_a: VectorData, vector_b: VectorData) -> Tuple[float, float]:
    """Get cached dot product and angle between vectors."""
    key = ("dot_angle", vector_a.id, vector_a.coords, vector_b.id, vector_b.coords)
    cached = _cache_get(_VECTOR_METRIC_CACHE, key)
    if cached is not None:
        return cached
    if len(vector_a.coords) != len(vector_b.coords):
        value = (0.0, 0.0)
        _cache_set(_VECTOR_METRIC_CACHE, key, value)
        return value
    dot = sum(a * b for a, b in zip(vector_a.coords, vector_b.coords))
    norm_a = sqrt(sum(a * a for a in vector_a.coords))
    norm_b = sqrt(sum(b * b for b in vector_b.coords))
    angle_deg = 0.0
    if norm_a > 1e-10 and norm_b > 1e-10:
        cos_angle = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
        angle_deg = degrees(acos(cos_angle))
    value = (float(dot), float(angle_deg))
    _cache_set(_VECTOR_METRIC_CACHE, key, value)
    return value


def get_vector_axis_projections(vector: VectorData) -> Tuple[float, float, float]:
    """Get cached projections onto axes."""
    key = ("proj", vector.id, vector.coords)
    cached = _cache_get(_VECTOR_METRIC_CACHE, key)
    if cached is not None:
        return cached
    coords = list(vector.coords)
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
