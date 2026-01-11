"""
Affine transform composition class.
"""

import numpy as np
from typing import Tuple, Optional

from domain.transforms.affine_matrices import (
    create_rotation_matrix,
    create_scale_matrix,
    create_translation_matrix,
    create_shear_matrix,
    create_flip_matrix,
)


class AffineTransform:
    """
    Represents a composable affine transformation.
    """

    def __init__(self, matrix: Optional[np.ndarray] = None):
        if matrix is None:
            self.matrix = np.eye(3, dtype=np.float32)
        else:
            self.matrix = np.array(matrix, dtype=np.float32)

    def rotate(self, angle: float, center: Optional[Tuple[float, float]] = None) -> 'AffineTransform':
        rot_matrix = create_rotation_matrix(angle, center)
        self.matrix = rot_matrix @ self.matrix
        return self

    def scale(self, sx: float, sy: Optional[float] = None,
              center: Optional[Tuple[float, float]] = None) -> 'AffineTransform':
        scale_matrix = create_scale_matrix(sx, sy, center)
        self.matrix = scale_matrix @ self.matrix
        return self

    def translate(self, tx: float, ty: float) -> 'AffineTransform':
        trans_matrix = create_translation_matrix(tx, ty)
        self.matrix = trans_matrix @ self.matrix
        return self

    def shear(self, shx: float, shy: float = 0) -> 'AffineTransform':
        shear_matrix = create_shear_matrix(shx, shy)
        self.matrix = shear_matrix @ self.matrix
        return self

    def flip_horizontal(self, width: float) -> 'AffineTransform':
        flip_matrix = create_flip_matrix(horizontal=True, width=width)
        self.matrix = flip_matrix @ self.matrix
        return self

    def flip_vertical(self, height: float) -> 'AffineTransform':
        flip_matrix = create_flip_matrix(vertical=True, height=height)
        self.matrix = flip_matrix @ self.matrix
        return self

    def compose(self, other: 'AffineTransform') -> 'AffineTransform':
        result = AffineTransform()
        result.matrix = self.matrix @ other.matrix
        return result

    def inverse(self) -> 'AffineTransform':
        result = AffineTransform()
        result.matrix = np.linalg.inv(self.matrix)
        return result

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        point = np.array([x, y, 1], dtype=np.float32)
        transformed = self.matrix @ point
        return (float(transformed[0]), float(transformed[1]))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        n = points.shape[0]
        homogeneous = np.hstack([points, np.ones((n, 1))])
        transformed = (self.matrix @ homogeneous.T).T
        return transformed[:, :2]

    def get_matrix_2x3(self) -> np.ndarray:
        return self.matrix[:2, :]

    def __str__(self) -> str:
        return f"AffineTransform:\n{self.matrix}"
