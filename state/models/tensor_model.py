"""
Unified Tensor data model for CVLA.

TensorData represents vectors, matrices, and images in a unified structure.
Semantic behavior is derived from rank and dtype.
"""

from dataclasses import dataclass, replace, field
from enum import Enum
from typing import Tuple, Union, Optional
from uuid import uuid4
import numpy as np
from state.input_parser import ParsedTensor


class TensorDType(Enum):
    """Data type classification for tensors."""
    NUMERIC = "numeric"
    IMAGE_RGB = "image_rgb"
    IMAGE_GRAYSCALE = "image_grayscale"


@dataclass(frozen=True)
class TensorData:
    """
    Unified immutable tensor representation.

    Can represent vectors, matrices, or images depending on shape and dtype.
    """
    id: str
    data: Tuple[Union[float, Tuple], ...]  # Nested tuples for N-dimensional
    shape: Tuple[int, ...]
    dtype: TensorDType
    label: str
    color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    visible: bool = True
    history: Tuple[str, ...] = ()
    original_data: Optional[Tuple[Union[float, Tuple], ...]] = None  # For reset functionality
    order: int = field(default=-1, repr=False)

    def __post_init__(self):
        """Infer order once at creation if not explicitly provided."""
        if self.order < 0:
            inferred = _infer_order_from_data(self.data)
            object.__setattr__(self, "order", inferred)

    @property
    def rank(self) -> int:
        """Number of dimensions."""
        return self.order

    @property
    def is_image_dtype(self) -> bool:
        return self.dtype in (TensorDType.IMAGE_RGB, TensorDType.IMAGE_GRAYSCALE)

    @staticmethod
    def create_vector(
        coords: Tuple[float, ...],
        label: str,
        color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    ) -> 'TensorData':
        """Create a vector tensor."""
        return TensorData(
            id=str(uuid4()),
            data=tuple(float(c) for c in coords),
            shape=(len(coords),),
            dtype=TensorDType.NUMERIC,
            label=label,
            color=color,
            visible=True,
            history=(),
            order=1
        )

    @staticmethod
    def create_matrix(
        values: Tuple[Tuple[float, ...], ...],
        label: str,
        color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    ) -> 'TensorData':
        """Create a matrix tensor."""
        normalized = _normalize_matrix_values(values)
        if not normalized:
            shape = (0, 0)
        else:
            shape = (len(normalized), len(normalized[0]))
        return TensorData(
            id=str(uuid4()),
            data=normalized,
            shape=shape,
            dtype=TensorDType.NUMERIC,
            label=label,
            color=color,
            visible=True,
            history=(),
            order=2
        )

    @staticmethod
    def from_parsed(
        parsed: ParsedTensor,
        label: str,
        color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    ) -> 'TensorData':
        """Create a tensor from a parsed tensor payload."""
        return TensorData(
            id=str(uuid4()),
            data=parsed.data,
            shape=parsed.shape,
            dtype=TensorDType.NUMERIC,
            label=label,
            color=color,
            visible=True,
            history=(),
            order=parsed.order
        )

    @staticmethod
    def create_image(
        pixels: np.ndarray,
        name: str,
        history: Tuple[str, ...] = (),
        preserve_original: bool = True
    ) -> 'TensorData':
        """
        Create an image tensor from numpy array.

        Args:
            pixels: HxW (grayscale) or HxWxC (color) numpy array
            name: Image label
            history: Operation history
            preserve_original: If True, store original data for reset functionality
        """
        shape = tuple(pixels.shape)
        is_grayscale = len(shape) == 2 or (len(shape) == 3 and shape[2] == 1)
        dtype = TensorDType.IMAGE_GRAYSCALE if is_grayscale else TensorDType.IMAGE_RGB

        # Convert numpy array to nested tuples
        data = _numpy_to_tuples(pixels)

        # Store original data for reset functionality (only for new images, not derived ones)
        original = data if (preserve_original and not history) else None

        return TensorData(
            id=str(uuid4()),
            data=data,
            shape=shape,
            dtype=dtype,
            label=name,
            color=(0.8, 0.8, 0.8),
            visible=True,
            history=history,
            original_data=original,
            order=len(shape)
        )

    def to_numpy(self) -> np.ndarray:
        """Convert data to numpy array."""
        return _tuples_to_numpy(self.data, self.shape)

    def with_history(self, operation: str) -> 'TensorData':
        """Return new TensorData with operation appended to history."""
        return replace(self, history=self.history + (operation,))

    def with_data(
        self,
        new_data: Tuple,
        new_shape: Optional[Tuple[int, ...]] = None,
        new_order: Optional[int] = None
    ) -> 'TensorData':
        """Return new TensorData with updated data."""
        shape = new_shape if new_shape is not None else self.shape
        return replace(
            self,
            data=new_data,
            shape=shape,
            order=new_order if new_order is not None else self.order
        )

    def with_label(self, label: str) -> 'TensorData':
        """Return new TensorData with updated label."""
        return replace(self, label=label)

    def with_color(self, color: Tuple[float, float, float]) -> 'TensorData':
        """Return new TensorData with updated color."""
        return replace(self, color=color)

    def with_visible(self, visible: bool) -> 'TensorData':
        """Return new TensorData with updated visibility."""
        return replace(self, visible=visible)

    # Vector-specific properties
    @property
    def coords(self) -> Tuple[float, ...]:
        """Get vector coordinates (only valid for vectors)."""
        if self.rank != 1:
            raise ValueError("coords only valid for vectors")
        return self.data

    # Matrix-specific properties
    @property
    def values(self) -> Tuple[Tuple[float, ...], ...]:
        """Get matrix values (only valid for matrices)."""
        if self.rank != 2:
            raise ValueError("values only valid for matrices")
        return self.data

    @property
    def rows(self) -> int:
        """Get number of rows (for matrices)."""
        if self.rank < 2:
            return 1
        return self.shape[0]

    @property
    def cols(self) -> int:
        """Get number of columns (for matrices)."""
        if self.rank < 2:
            return self.shape[0] if self.shape else 0
        return self.shape[1]

    # Image-specific properties
    @property
    def height(self) -> int:
        """Get image height (only valid for images)."""
        if not self.is_image_dtype:
            raise ValueError("height only valid for images")
        return self.shape[0]

    @property
    def width(self) -> int:
        """Get image width (only valid for images)."""
        if not self.is_image_dtype:
            raise ValueError("width only valid for images")
        return self.shape[1]

    @property
    def channels(self) -> int:
        """Get number of channels (only valid for images)."""
        if not self.is_image_dtype:
            raise ValueError("channels only valid for images")
        return self.shape[2] if len(self.shape) > 2 else 1

    @property
    def is_grayscale(self) -> bool:
        """Check if image is grayscale."""
        return self.dtype == TensorDType.IMAGE_GRAYSCALE


def _numpy_to_tuples(arr: np.ndarray) -> Tuple:
    """Convert numpy array to nested tuples."""
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        return tuple(float(x) for x in arr)
    return tuple(_numpy_to_tuples(row) for row in arr)


def _tuples_to_numpy(data: Tuple, shape: Tuple[int, ...]) -> np.ndarray:
    """Convert nested tuples back to numpy array."""
    arr = np.array(data, dtype=np.float32)
    return arr.reshape(shape) if arr.shape != shape else arr


def _normalize_matrix_values(values: Tuple[Tuple[float, ...], ...]) -> Tuple[Tuple[float, ...], ...]:
    """Ensure matrix rows are rectangular by padding with zeros."""
    if not values:
        return ()
    rows = [tuple(float(v) for v in row) for row in values if row is not None]
    if not rows:
        return ()
    max_cols = max((len(row) for row in rows), default=0)
    if max_cols == 0:
        return ()
    normalized = []
    for row in rows:
        if len(row) < max_cols:
            row = row + (0.0,) * (max_cols - len(row))
        elif len(row) > max_cols:
            row = row[:max_cols]
        normalized.append(row)
    return tuple(normalized)


def _infer_order_from_data(data) -> int:
    """Infer tensor order from nested tuple data once at creation."""
    if isinstance(data, (int, float)):
        return 0
    if isinstance(data, tuple):
        if not data:
            return 1
        first_order = _infer_order_from_data(data[0])
        for item in data:
            if _infer_order_from_data(item) != first_order:
                raise ValueError("Tensor data must be uniform across dimensions.")
        return first_order + 1
    raise ValueError("Unsupported tensor data type for order inference.")
