"""
Affine transform helpers for visualization and augmentation.
"""

import numpy as np
from typing import Tuple, Union

from vision.affine_transform import AffineTransform


def create_random_augmentation() -> AffineTransform:
    """
    Create a random augmentation transform.
    """
    transform = AffineTransform()

    angle = np.random.uniform(-15, 15)
    transform.rotate(angle)

    scale = np.random.uniform(0.9, 1.1)
    transform.scale(scale)

    tx = np.random.uniform(-0.1, 0.1)
    ty = np.random.uniform(-0.1, 0.1)
    transform.translate(tx, ty)

    return transform


def visualize_transformation_grid(transform: Union[np.ndarray, AffineTransform],
                                  size: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a grid visualization showing how transformation warps space.
    """
    if isinstance(transform, AffineTransform):
        t = transform
    else:
        t = AffineTransform(transform)

    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)

    original = np.column_stack([xx.ravel(), yy.ravel()])
    transformed = t.transform_points(original)

    return original, transformed
