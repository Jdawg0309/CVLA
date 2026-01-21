"""
Tensor actions for the unified tensor model.

These actions handle CRUD operations on tensors and operation application.
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass(frozen=True)
class AddTensor:
    """Add a new tensor to the scene."""
    data: Tuple
    shape: Tuple[int, ...]
    dtype: str  # "numeric", "image_rgb", "image_grayscale"
    label: str
    color: Tuple[float, float, float] = (0.8, 0.8, 0.8)


@dataclass(frozen=True)
class DeleteTensor:
    """Delete a tensor by ID."""
    id: str


@dataclass(frozen=True)
class UpdateTensor:
    """Update tensor properties."""
    id: str
    data: Optional[Tuple] = None
    shape: Optional[Tuple[int, ...]] = None
    label: Optional[str] = None
    color: Optional[Tuple[float, float, float]] = None
    visible: Optional[bool] = None


@dataclass(frozen=True)
class SelectTensor:
    """Select a tensor by ID."""
    id: str


@dataclass(frozen=True)
class DeselectTensor:
    """Clear tensor selection."""
    pass


@dataclass(frozen=True)
class ApplyOperation:
    """
    Apply an operation to one or more tensors.

    Args:
        operation_name: Name of the operation (e.g., "normalize", "scale", "apply_kernel")
        parameters: Key-value pairs as tuples
        target_ids: IDs of tensors to operate on
        create_new: If True, create new tensor(s); if False, modify in place
    """
    operation_name: str
    parameters: Tuple[Tuple[str, str], ...]
    target_ids: Tuple[str, ...]
    create_new: bool = True


@dataclass(frozen=True)
class PreviewOperation:
    """
    Preview an operation result without applying.

    Args:
        operation_name: Name of the operation
        parameters: Key-value pairs as tuples
        target_id: ID of tensor to preview operation on
    """
    operation_name: str
    parameters: Tuple[Tuple[str, str], ...]
    target_id: str


@dataclass(frozen=True)
class CancelPreview:
    """Cancel the current operation preview."""
    pass


@dataclass(frozen=True)
class ConfirmPreview:
    """Confirm and apply the previewed operation."""
    pass


@dataclass(frozen=True)
class SetBinaryOperation:
    """Set a pending binary operation waiting for second tensor."""
    operation_name: str
    first_tensor_id: str


@dataclass(frozen=True)
class ClearBinaryOperation:
    """Clear the pending binary operation."""
    pass


@dataclass(frozen=True)
class ClearAllTensors:
    """Clear all tensors from the scene."""
    pass


@dataclass(frozen=True)
class DuplicateTensor:
    """Duplicate an existing tensor."""
    id: str
    new_label: Optional[str] = None


# Convenience actions for creating specific tensor types

@dataclass(frozen=True)
class AddVectorTensor:
    """Add a vector tensor (convenience action)."""
    coords: Tuple[float, ...]
    label: str
    color: Tuple[float, float, float] = (0.8, 0.8, 0.8)


@dataclass(frozen=True)
class AddMatrixTensor:
    """Add a matrix tensor (convenience action)."""
    values: Tuple[Tuple[float, ...], ...]
    label: str
    color: Tuple[float, float, float] = (0.8, 0.8, 0.8)


@dataclass(frozen=True)
class AddImageTensor:
    """Add an image tensor from file or sample."""
    source: str  # "file" or "sample"
    path: str = ""  # File path if source is "file"
    pattern: str = "checkerboard"  # Pattern name if source is "sample"
    size: int = 64  # Size if source is "sample"
    label: str = ""
