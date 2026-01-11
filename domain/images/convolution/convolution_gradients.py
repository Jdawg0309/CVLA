"""
Gradient utilities for convolution outputs.
"""

import numpy as np

from domain.images.convolution.convolution_core import convolve2d
from domain.images.image import ImageMatrix
from domain.images.kernels.kernels import SOBEL_X, SOBEL_Y


def compute_gradient_magnitude(image: ImageMatrix) -> np.ndarray:
    """Compute gradient magnitude using Sobel operators."""
    data = image.as_matrix()
    gx = convolve2d(data, SOBEL_X)
    gy = convolve2d(data, SOBEL_Y)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()

    return magnitude


def compute_gradient_direction(image: ImageMatrix) -> np.ndarray:
    """Compute gradient direction using Sobel operators."""
    data = image.as_matrix()
    gx = convolve2d(data, SOBEL_X)
    gy = convolve2d(data, SOBEL_Y)

    return np.arctan2(gy, gx)
