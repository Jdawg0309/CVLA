"""
Cached image-derived values for reducers.
"""

from typing import Optional, Tuple

import numpy as np

from state.models import ImageData


def _as_grayscale_matrix(image: ImageData) -> np.ndarray:
    if image.is_grayscale:
        if len(image.pixels.shape) == 2:
            return image.pixels
        return image.pixels[:, :, 0]
    return (
        0.299 * image.pixels[:, :, 0]
        + 0.587 * image.pixels[:, :, 1]
        + 0.114 * image.pixels[:, :, 2]
    )


def compute_image_stats(image: Optional[ImageData]) -> Optional[Tuple[float, float, float, float]]:
    if image is None:
        return None
    matrix = _as_grayscale_matrix(image)
    mean = float(np.mean(matrix))
    std = float(np.std(matrix))
    min_val = float(np.min(matrix))
    max_val = float(np.max(matrix))
    return (mean, std, min_val, max_val)


def compute_preview_matrix(
    image: Optional[ImageData],
    max_rows: int = 8,
    max_cols: int = 8,
) -> Optional[Tuple[Tuple[float, ...], ...]]:
    if image is None:
        return None
    matrix = _as_grayscale_matrix(image)
    rows = min(max_rows, matrix.shape[0])
    cols = min(max_cols, matrix.shape[1])
    preview = tuple(
        tuple(float(matrix[r, c]) for c in range(cols))
        for r in range(rows)
    )
    return preview
