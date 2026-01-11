"""
Affine transformation matrix helpers.
"""

import numpy as np
from typing import Tuple, Optional


def create_rotation_matrix(angle: float, center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    theta = np.radians(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    rotation = np.array([
        [cos_t, -sin_t, 0],
        [sin_t, cos_t, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    if center is not None:
        cx, cy = center
        to_origin = create_translation_matrix(-cx, -cy)
        from_origin = create_translation_matrix(cx, cy)
        return from_origin @ rotation @ to_origin

    return rotation


def create_scale_matrix(sx: float, sy: Optional[float] = None,
                        center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    if sy is None:
        sy = sx

    scale = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    if center is not None:
        cx, cy = center
        to_origin = create_translation_matrix(-cx, -cy)
        from_origin = create_translation_matrix(cx, cy)
        return from_origin @ scale @ to_origin

    return scale


def create_translation_matrix(tx: float, ty: float) -> np.ndarray:
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float32)


def create_shear_matrix(shx: float, shy: float = 0) -> np.ndarray:
    return np.array([
        [1, shx, 0],
        [shy, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)


def create_flip_matrix(horizontal: bool = False, vertical: bool = False,
                       width: float = 1, height: float = 1) -> np.ndarray:
    sx = -1 if horizontal else 1
    sy = -1 if vertical else 1

    flip = np.array([
        [sx, 0, width if horizontal else 0],
        [0, sy, height if vertical else 0],
        [0, 0, 1]
    ], dtype=np.float32)

    return flip
