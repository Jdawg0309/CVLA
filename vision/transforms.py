"""
Affine Transformations for Images

This module implements geometric transformations on images using matrices.
These transformations demonstrate how linear algebra operations on coordinates
produce visual changes in images.

ML/CV Insight:
    In computer vision and neural networks, spatial transformations are used for:
    - Data augmentation (random rotations, scaling, flips during training)
    - Spatial Transformer Networks (learning where to look in an image)
    - Image alignment and registration
    - Perspective correction

Mathematical Foundation:
    Affine transformations can be represented as 3x3 matrices operating on
    homogeneous coordinates [x, y, 1]:

    [x']   [a b tx]   [x]
    [y'] = [c d ty] * [y]
    [1 ]   [0 0 1 ]   [1]

    This single matrix can encode rotation, scaling, shearing, and translation.
"""

import numpy as np
from typing import Tuple, Optional, Union
from scipy import ndimage

from .image import ImageMatrix


def create_rotation_matrix(angle: float, center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Create a 3x3 rotation matrix.

    Args:
        angle: Rotation angle in degrees (counter-clockwise)
        center: Optional rotation center (x, y). If None, rotates around origin.

    Returns:
        3x3 transformation matrix

    ML/CV Insight:
        Rotation is a fundamental transformation for data augmentation.
        Networks trained with random rotations become rotation-invariant.

    Mathematical Form:
        R = [cos(θ)  -sin(θ)  0]
            [sin(θ)   cos(θ)  0]
            [0        0       1]
    """
    theta = np.radians(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Basic rotation matrix
    rotation = np.array([
        [cos_t, -sin_t, 0],
        [sin_t, cos_t, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    if center is not None:
        cx, cy = center
        # Translate to origin, rotate, translate back
        to_origin = create_translation_matrix(-cx, -cy)
        from_origin = create_translation_matrix(cx, cy)
        return from_origin @ rotation @ to_origin

    return rotation


def create_scale_matrix(sx: float, sy: Optional[float] = None,
                        center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Create a 3x3 scaling matrix.

    Args:
        sx: Scale factor in x direction
        sy: Scale factor in y direction (defaults to sx for uniform scaling)
        center: Optional scale center. If None, scales from origin.

    Returns:
        3x3 transformation matrix

    ML/CV Insight:
        Scaling is used for:
        - Multi-scale feature extraction (image pyramids)
        - Data augmentation
        - Resizing inputs to fixed network dimensions

    Mathematical Form:
        S = [sx  0   0]
            [0   sy  0]
            [0   0   1]
    """
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
    """
    Create a 3x3 translation matrix.

    Args:
        tx: Translation in x direction
        ty: Translation in y direction

    Returns:
        3x3 transformation matrix

    ML/CV Insight:
        Translation shifts the image. Combined with cropping, this is
        used for data augmentation (random crops) in training.

    Mathematical Form:
        T = [1  0  tx]
            [0  1  ty]
            [0  0  1 ]
    """
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float32)


def create_shear_matrix(shx: float, shy: float = 0) -> np.ndarray:
    """
    Create a 3x3 shear matrix.

    Args:
        shx: Shear factor in x direction (horizontal shear)
        shy: Shear factor in y direction (vertical shear)

    Returns:
        3x3 transformation matrix

    ML/CV Insight:
        Shearing skews the image. While less common than rotation/scaling,
        it's part of the full affine transformation family.

    Mathematical Form:
        Sh = [1    shx  0]
             [shy  1    0]
             [0    0    1]
    """
    return np.array([
        [1, shx, 0],
        [shy, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)


def create_flip_matrix(horizontal: bool = False, vertical: bool = False,
                       width: float = 1, height: float = 1) -> np.ndarray:
    """
    Create a flip/mirror transformation matrix.

    Args:
        horizontal: If True, flip horizontally
        vertical: If True, flip vertically
        width: Image width (needed to keep image in view after flip)
        height: Image height

    Returns:
        3x3 transformation matrix

    ML/CV Insight:
        Horizontal flipping is extremely common data augmentation.
        It doubles the effective dataset size with minimal computation.
        Most objects look valid when flipped (except text, etc.).
    """
    sx = -1 if horizontal else 1
    sy = -1 if vertical else 1

    flip = np.array([
        [sx, 0, width if horizontal else 0],
        [0, sy, height if vertical else 0],
        [0, 0, 1]
    ], dtype=np.float32)

    return flip


class AffineTransform:
    """
    Represents a composable affine transformation.

    This class allows chaining multiple transformations together,
    demonstrating how matrix multiplication combines transformations.

    ML/CV Insight:
        Spatial Transformer Networks (STNs) learn these transformation
        parameters from data. The network predicts transformation matrices
        that align input images to canonical poses.
    """

    def __init__(self, matrix: Optional[np.ndarray] = None):
        """
        Initialize with optional transformation matrix.

        Args:
            matrix: 3x3 transformation matrix. If None, starts with identity.
        """
        if matrix is None:
            self.matrix = np.eye(3, dtype=np.float32)
        else:
            self.matrix = np.array(matrix, dtype=np.float32)

    def rotate(self, angle: float, center: Optional[Tuple[float, float]] = None) -> 'AffineTransform':
        """Add rotation to the transformation chain."""
        rot_matrix = create_rotation_matrix(angle, center)
        self.matrix = rot_matrix @ self.matrix
        return self

    def scale(self, sx: float, sy: Optional[float] = None,
              center: Optional[Tuple[float, float]] = None) -> 'AffineTransform':
        """Add scaling to the transformation chain."""
        scale_matrix = create_scale_matrix(sx, sy, center)
        self.matrix = scale_matrix @ self.matrix
        return self

    def translate(self, tx: float, ty: float) -> 'AffineTransform':
        """Add translation to the transformation chain."""
        trans_matrix = create_translation_matrix(tx, ty)
        self.matrix = trans_matrix @ self.matrix
        return self

    def shear(self, shx: float, shy: float = 0) -> 'AffineTransform':
        """Add shearing to the transformation chain."""
        shear_matrix = create_shear_matrix(shx, shy)
        self.matrix = shear_matrix @ self.matrix
        return self

    def flip_horizontal(self, width: float) -> 'AffineTransform':
        """Add horizontal flip."""
        flip_matrix = create_flip_matrix(horizontal=True, width=width)
        self.matrix = flip_matrix @ self.matrix
        return self

    def flip_vertical(self, height: float) -> 'AffineTransform':
        """Add vertical flip."""
        flip_matrix = create_flip_matrix(vertical=True, height=height)
        self.matrix = flip_matrix @ self.matrix
        return self

    def compose(self, other: 'AffineTransform') -> 'AffineTransform':
        """Compose with another transformation (this after other)."""
        result = AffineTransform()
        result.matrix = self.matrix @ other.matrix
        return result

    def inverse(self) -> 'AffineTransform':
        """
        Get the inverse transformation.

        ML/CV Insight:
            The inverse transformation maps output coordinates back to input.
            This is how we actually apply transformations to images -
            for each output pixel, we find where it came from in the input.
        """
        result = AffineTransform()
        result.matrix = np.linalg.inv(self.matrix)
        return result

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """Transform a single point."""
        point = np.array([x, y, 1], dtype=np.float32)
        transformed = self.matrix @ point
        return (float(transformed[0]), float(transformed[1]))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform multiple points.

        Args:
            points: Nx2 array of (x, y) coordinates

        Returns:
            Nx2 array of transformed coordinates
        """
        n = points.shape[0]
        homogeneous = np.hstack([points, np.ones((n, 1))])
        transformed = (self.matrix @ homogeneous.T).T
        return transformed[:, :2]

    def get_matrix_2x3(self) -> np.ndarray:
        """Get the 2x3 matrix for OpenCV-style transforms."""
        return self.matrix[:2, :]

    def __str__(self) -> str:
        return f"AffineTransform:\n{self.matrix}"


def apply_affine_transform(image_matrix: ImageMatrix,
                           transform: Union[np.ndarray, AffineTransform],
                           output_shape: Optional[Tuple[int, int]] = None,
                           order: int = 1) -> ImageMatrix:
    """
    Apply an affine transformation to an image.

    Args:
        image_matrix: Input ImageMatrix
        transform: 3x3 transformation matrix or AffineTransform object
        output_shape: (height, width) of output. If None, same as input.
        order: Interpolation order (0=nearest, 1=bilinear, 3=cubic)

    Returns:
        Transformed ImageMatrix

    ML/CV Insight:
        Image transformation involves:
        1. For each output pixel, compute where it maps in input (inverse transform)
        2. Sample the input at that location using interpolation
        3. Bilinear interpolation (order=1) is common - it's differentiable!

        This is why Spatial Transformer Networks can be trained end-to-end:
        the sampling operation is differentiable.
    """
    if isinstance(transform, AffineTransform):
        matrix = transform.matrix
    else:
        matrix = np.array(transform, dtype=np.float32)

    data = image_matrix.data
    is_color = len(data.shape) == 3

    if output_shape is None:
        output_shape = data.shape[:2]

    # We need to use the inverse transform for mapping
    # (for each output pixel, where does it come from in input?)
    inv_matrix = np.linalg.inv(matrix)

    if is_color:
        # Transform each channel
        output = np.zeros((*output_shape, data.shape[2]), dtype=np.float32)
        for c in range(data.shape[2]):
            output[:, :, c] = ndimage.affine_transform(
                data[:, :, c],
                inv_matrix[:2, :2],
                offset=inv_matrix[:2, 2],
                output_shape=output_shape,
                order=order,
                mode='constant',
                cval=0
            )
    else:
        output = ndimage.affine_transform(
            data,
            inv_matrix[:2, :2],
            offset=inv_matrix[:2, 2],
            output_shape=output_shape,
            order=order,
            mode='constant',
            cval=0
        )

    result = ImageMatrix(output, f"{image_matrix.name}_transformed")
    result.history = image_matrix.history.copy()
    result.history.append(('affine_transform', matrix.tolist()))
    return result


def normalize_image(image_matrix: ImageMatrix,
                    mean: Optional[Union[float, Tuple[float, ...]]] = None,
                    std: Optional[Union[float, Tuple[float, ...]]] = None) -> ImageMatrix:
    """
    Normalize image data (subtract mean, divide by std).

    Args:
        image_matrix: Input ImageMatrix
        mean: Mean to subtract. If None, compute from image.
        std: Std to divide by. If None, compute from image.

    Returns:
        Normalized ImageMatrix

    ML/CV Insight:
        Normalization is CRITICAL for neural networks:
        - Puts all pixels on similar scale
        - Helps gradient flow during training
        - ImageNet models use specific mean/std values:
          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

        This is different from batch normalization (which normalizes
        activations during training) but serves similar purposes.
    """
    data = image_matrix.data.copy()

    if mean is None:
        mean = np.mean(data)
    if std is None:
        std = np.std(data)
        if std < 1e-8:
            std = 1.0

    normalized = (data - mean) / std

    result = ImageMatrix(normalized, f"{image_matrix.name}_normalized")
    result.history = image_matrix.history.copy()
    result.history.append(('normalize', {'mean': mean, 'std': std}))
    return result


def create_random_augmentation() -> AffineTransform:
    """
    Create a random augmentation transform.

    Returns:
        Random AffineTransform

    ML/CV Insight:
        Random augmentation is THE most important technique for preventing
        overfitting in image classification. It creates "virtual" training
        examples by applying random transformations.

        Common augmentations:
        - Random rotation: +/- 15 degrees
        - Random scale: 0.9 to 1.1
        - Random translation: +/- 10% of image size
        - Random horizontal flip
    """
    transform = AffineTransform()

    # Random rotation (-15 to +15 degrees)
    angle = np.random.uniform(-15, 15)
    transform.rotate(angle)

    # Random scale (0.9 to 1.1)
    scale = np.random.uniform(0.9, 1.1)
    transform.scale(scale)

    # Random translation (-0.1 to +0.1)
    tx = np.random.uniform(-0.1, 0.1)
    ty = np.random.uniform(-0.1, 0.1)
    transform.translate(tx, ty)

    return transform


def visualize_transformation_grid(transform: Union[np.ndarray, AffineTransform],
                                  size: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a grid visualization showing how transformation warps space.

    Args:
        transform: Transformation to visualize
        size: Grid size

    Returns:
        (original_points, transformed_points) as Nx2 arrays

    ML/CV Insight:
        Visualizing the transformation grid shows intuitively how
        different transformations warp the coordinate space.
    """
    if isinstance(transform, AffineTransform):
        t = transform
    else:
        t = AffineTransform(transform)

    # Create grid points
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)

    original = np.column_stack([xx.ravel(), yy.ravel()])
    transformed = t.transform_points(original)

    return original, transformed
