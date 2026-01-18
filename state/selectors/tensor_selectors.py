"""
Tensor selectors for querying tensor state.

These functions provide efficient access to tensor data from the app state.
"""

from typing import Optional, Tuple, List, TYPE_CHECKING
from collections import OrderedDict

if TYPE_CHECKING:
    from state.app_state import AppState
    from state.models.tensor_model import TensorData
from state.models.tensor_model import TensorDType


# Cache for computed values
_TENSOR_CACHE = OrderedDict()
_CACHE_MAX_SIZE = 256


def _cache_get(key):
    """Get value from cache with LRU behavior."""
    if key in _TENSOR_CACHE:
        value = _TENSOR_CACHE.pop(key)
        _TENSOR_CACHE[key] = value
        return value
    return None


def _cache_set(key, value):
    """Set value in cache with LRU eviction."""
    _TENSOR_CACHE[key] = value
    if len(_TENSOR_CACHE) > _CACHE_MAX_SIZE:
        _TENSOR_CACHE.popitem(last=False)


def get_tensor_by_id(state: "AppState", tensor_id: str) -> Optional["TensorData"]:
    """Find a tensor by ID."""
    for t in state.tensors:
        if t.id == tensor_id:
            return t
    return None


def get_selected_tensor(state: "AppState") -> Optional["TensorData"]:
    """Get the currently selected tensor, if any."""
    if state.selected_tensor_id:
        return get_tensor_by_id(state, state.selected_tensor_id)
    return None


def get_tensors_by_type(state: "AppState", tensor_type: str) -> Tuple["TensorData", ...]:
    """
    Get all tensors of a specific type.

    Args:
        state: App state
        tensor_type: 'vector', 'matrix', or 'image'

    Returns:
        Tuple of matching tensors
    """
    if tensor_type == "vector":
        return tuple(t for t in state.tensors if t.rank == 1)
    if tensor_type == "matrix":
        return tuple(t for t in state.tensors if t.rank == 2)
    if tensor_type == "image":
        return tuple(t for t in state.tensors if t.dtype in (TensorDType.IMAGE_RGB, TensorDType.IMAGE_GRAYSCALE))
    return ()


def get_vectors(state: "AppState") -> Tuple["TensorData", ...]:
    """Get all vector tensors (rank-1)."""
    return get_tensors_by_type(state, 'vector')


def get_matrices(state: "AppState") -> Tuple["TensorData", ...]:
    """Get all matrix tensors (rank-2, non-image)."""
    return tuple(
        t for t in state.tensors if t.rank == 2 and not t.is_image_dtype
    )


def get_images(state: "AppState") -> Tuple["TensorData", ...]:
    """Get all image tensors."""
    return get_tensors_by_type(state, 'image')


def get_visible_tensors(state: "AppState") -> Tuple["TensorData", ...]:
    """Get all visible tensors."""
    return tuple(t for t in state.tensors if t.visible)


def get_visible_vectors(state: "AppState") -> Tuple["TensorData", ...]:
    """Get all visible vector tensors."""
    return tuple(t for t in state.tensors if t.rank == 1 and t.visible)


def get_visible_matrices(state: "AppState") -> Tuple["TensorData", ...]:
    """Get all visible matrix tensors."""
    return tuple(t for t in state.tensors if t.rank == 2 and t.visible)


def get_visible_images(state: "AppState") -> Tuple["TensorData", ...]:
    """Get all visible image tensors."""
    return tuple(
        t for t in state.tensors
        if t.dtype in (TensorDType.IMAGE_RGB, TensorDType.IMAGE_GRAYSCALE) and t.visible
    )


def get_tensor_count(state: "AppState") -> int:
    """Get total number of tensors."""
    return len(state.tensors)


def get_tensor_count_by_type(state: "AppState", tensor_type: str) -> int:
    """Get count of tensors of a specific type."""
    return len(get_tensors_by_type(state, tensor_type))


def get_tensor_labels(state: "AppState") -> Tuple[str, ...]:
    """Get all tensor labels."""
    return tuple(t.label for t in state.tensors)


def get_tensor_ids(state: "AppState") -> Tuple[str, ...]:
    """Get all tensor IDs."""
    return tuple(t.id for t in state.tensors)


def is_tensor_selected(state: "AppState", tensor_id: str) -> bool:
    """Check if a specific tensor is selected."""
    return state.selected_tensor_id == tensor_id


def has_tensors(state: "AppState") -> bool:
    """Check if there are any tensors."""
    return len(state.tensors) > 0


def has_vectors(state: "AppState") -> bool:
    """Check if there are any vector tensors."""
    return any(t.rank == 1 for t in state.tensors)


def has_matrices(state: "AppState") -> bool:
    """Check if there are any matrix tensors."""
    return any(t.rank == 2 for t in state.tensors)


def has_images(state: "AppState") -> bool:
    """Check if there are any image tensors."""
    return any(t.dtype in (TensorDType.IMAGE_RGB, TensorDType.IMAGE_GRAYSCALE) for t in state.tensors)


# Computed properties with caching

def get_tensor_magnitude(tensor: "TensorData") -> float:
    """Get magnitude for vector tensors (cached)."""
    if tensor.rank != 1:
        return 0.0

    key = ("mag", tensor.id, tensor.data)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    from math import sqrt
    coords = tensor.data
    value = sqrt(sum(c * c for c in coords))
    _cache_set(key, value)
    return value


def get_tensor_norm(tensor: "TensorData") -> float:
    """Get Frobenius norm for matrix tensors (cached)."""
    if tensor.rank != 2:
        return get_tensor_magnitude(tensor) if tensor.rank == 1 else 0.0

    key = ("norm", tensor.id, tensor.data)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    from math import sqrt
    total = 0.0
    for row in tensor.data:
        for val in row:
            total += val * val
    value = sqrt(total)
    _cache_set(key, value)
    return value


def get_tensor_stats(tensor: "TensorData") -> Tuple[float, float, float, float]:
    """
    Get statistics for a tensor (cached).

    Returns:
        (min, max, mean, std) tuple
    """
    key = ("stats", tensor.id, tensor.data)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    import numpy as np
    arr = tensor.to_numpy()
    value = (
        float(np.min(arr)),
        float(np.max(arr)),
        float(np.mean(arr)),
        float(np.std(arr))
    )
    _cache_set(key, value)
    return value


def get_operation_history(state: "AppState") -> Tuple:
    """Get the operation history."""
    return state.operation_history


def get_last_operation(state: "AppState"):
    """Get the most recent operation record."""
    if state.operation_history:
        return state.operation_history[-1]
    return None


def get_preview_tensor(state: "AppState") -> Optional["TensorData"]:
    """Get the current operation preview tensor."""
    return state.operation_preview_tensor


def has_preview(state: "AppState") -> bool:
    """Check if there's an active operation preview."""
    return state.operation_preview_tensor is not None


def get_pending_operation(state: "AppState") -> Optional[str]:
    """Get the pending operation name."""
    return state.pending_operation


# Input panel selectors

def get_active_input_method(state: "AppState") -> str:
    """Get the active input method."""
    return state.active_input_method


def get_text_input_content(state: "AppState") -> str:
    """Get the current text input content."""
    return state.input_text_content


def get_text_input_type(state: "AppState") -> str:
    """Get the parsed type of text input."""
    return state.input_text_parsed_type


def get_grid_size(state: "AppState") -> Tuple[int, int]:
    """Get the grid size as (rows, cols)."""
    return (state.input_grid_rows, state.input_grid_cols)


def get_grid_cells(state: "AppState") -> Tuple[Tuple[float, ...], ...]:
    """Get the grid cell values."""
    return state.input_grid_cells


def get_file_path(state: "AppState") -> str:
    """Get the current file input path."""
    return state.input_file_path


def can_create_tensor_from_text(state: "AppState") -> bool:
    """Check if a tensor can be created from current text input."""
    return state.input_text_parsed_type in ("vector", "matrix")


def can_create_tensor_from_file(state: "AppState") -> bool:
    """Check if a tensor can be created from current file input."""
    return bool(state.input_file_path)


def can_create_tensor_from_grid(state: "AppState") -> bool:
    """Check if a tensor can be created from current grid input."""
    return state.input_grid_rows > 0 and state.input_grid_cols > 0
