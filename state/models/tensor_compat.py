"""
Compatibility layer for converting between legacy models and TensorData.

This module provides conversion functions to migrate existing VectorData,
MatrixData, and ImageData instances to the unified TensorData model.
"""

from typing import Optional
import numpy as np

from state.models.vector_model import VectorData
from state.models.matrix_model import MatrixData
from state.models.image_model import ImageData
from state.models.tensor_model import TensorData, TensorDType


def vector_to_tensor(vector: VectorData) -> TensorData:
    """
    Convert a VectorData instance to TensorData.

    Args:
        vector: Legacy VectorData instance

    Returns:
        Equivalent TensorData instance
    """
    return TensorData(
        id=vector.id,
        data=tuple(float(c) for c in vector.coords),
        shape=(len(vector.coords),),
        dtype=TensorDType.NUMERIC,
        label=vector.label,
        color=vector.color,
        visible=vector.visible,
        history=()
    )


def tensor_to_vector(tensor: TensorData) -> Optional[VectorData]:
    """
    Convert a TensorData instance back to VectorData.

    Args:
        tensor: TensorData instance (must be a vector type)

    Returns:
        Equivalent VectorData instance, or None if not a vector
    """
    if not tensor.is_vector:
        return None

    return VectorData(
        id=tensor.id,
        coords=tensor.data,
        color=tensor.color,
        label=tensor.label,
        visible=tensor.visible
    )


def matrix_to_tensor(matrix: MatrixData) -> TensorData:
    """
    Convert a MatrixData instance to TensorData.

    Args:
        matrix: Legacy MatrixData instance

    Returns:
        Equivalent TensorData instance
    """
    return TensorData(
        id=matrix.id,
        data=matrix.values,
        shape=matrix.shape,
        dtype=TensorDType.NUMERIC,
        label=matrix.label,
        color=(0.8, 0.8, 0.8),
        visible=matrix.visible,
        history=()
    )


def tensor_to_matrix(tensor: TensorData) -> Optional[MatrixData]:
    """
    Convert a TensorData instance back to MatrixData.

    Args:
        tensor: TensorData instance (must be a matrix type)

    Returns:
        Equivalent MatrixData instance, or None if not a matrix
    """
    if not tensor.is_matrix:
        return None

    return MatrixData(
        id=tensor.id,
        values=tensor.data,
        label=tensor.label,
        visible=tensor.visible
    )


def image_to_tensor(image: ImageData) -> TensorData:
    """
    Convert an ImageData instance to TensorData.

    Args:
        image: Legacy ImageData instance

    Returns:
        Equivalent TensorData instance
    """
    shape = tuple(image.pixels.shape)
    is_grayscale = image.is_grayscale
    dtype = TensorDType.IMAGE_GRAYSCALE if is_grayscale else TensorDType.IMAGE_RGB

    # Convert numpy array to nested tuples
    data = _numpy_to_tuples(image.pixels)

    return TensorData(
        id=image.id,
        data=data,
        shape=shape,
        dtype=dtype,
        label=image.name,
        color=(0.8, 0.8, 0.8),
        visible=True,
        history=image.history
    )


def tensor_to_image(tensor: TensorData) -> Optional[ImageData]:
    """
    Convert a TensorData instance back to ImageData.

    Args:
        tensor: TensorData instance (must be an image type)

    Returns:
        Equivalent ImageData instance, or None if not an image
    """
    if not tensor.is_image:
        return None

    pixels = tensor.to_numpy()

    return ImageData(
        id=tensor.id,
        pixels=pixels,
        name=tensor.label,
        shape=tensor.shape,
        history=tensor.history
    )


def _numpy_to_tuples(arr: np.ndarray) -> tuple:
    """Convert numpy array to nested tuples."""
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        return tuple(float(x) for x in arr)
    return tuple(_numpy_to_tuples(row) for row in arr)


# Utility functions for batch conversions

def vectors_to_tensors(vectors: tuple) -> tuple:
    """Convert a tuple of VectorData to a tuple of TensorData."""
    return tuple(vector_to_tensor(v) for v in vectors)


def matrices_to_tensors(matrices: tuple) -> tuple:
    """Convert a tuple of MatrixData to a tuple of TensorData."""
    return tuple(matrix_to_tensor(m) for m in matrices)


def tensors_to_vectors(tensors: tuple) -> tuple:
    """Extract vectors from a tuple of TensorData."""
    result = []
    for t in tensors:
        v = tensor_to_vector(t)
        if v is not None:
            result.append(v)
    return tuple(result)


def tensors_to_matrices(tensors: tuple) -> tuple:
    """Extract matrices from a tuple of TensorData."""
    result = []
    for t in tensors:
        m = tensor_to_matrix(t)
        if m is not None:
            result.append(m)
    return tuple(result)


def filter_tensors_by_type(tensors: tuple, tensor_type: str) -> tuple:
    """
    Filter tensors by type.

    Args:
        tensors: Tuple of TensorData
        tensor_type: 'vector', 'matrix', or 'image'

    Returns:
        Filtered tuple of TensorData
    """
    return tuple(t for t in tensors if t.tensor_type == tensor_type)
