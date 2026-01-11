"""
Image transformation operations.
"""

import numpy as np
from typing import Tuple, Optional, Union
from scipy import ndimage

from vision.affine_transform import AffineTransform
from vision.image_matrix import ImageMatrix


def apply_affine_transform(image_matrix: ImageMatrix,
                           transform: Union[np.ndarray, AffineTransform],
                           output_shape: Optional[Tuple[int, int]] = None,
                           order: int = 1) -> ImageMatrix:
    """
    Apply an affine transformation to an image.
    """
    if isinstance(transform, AffineTransform):
        matrix = transform.matrix
    else:
        matrix = np.array(transform, dtype=np.float32)

    data = image_matrix.data
    is_color = len(data.shape) == 3

    if output_shape is None:
        output_shape = data.shape[:2]

    inv_matrix = np.linalg.inv(matrix)

    if is_color:
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
