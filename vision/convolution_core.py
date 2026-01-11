"""
Core convolution operations.
"""

import numpy as np
from scipy import signal

from vision.image import ImageMatrix
from vision.kernels import get_kernel_by_name


def convolve2d(image: np.ndarray,
               kernel: np.ndarray,
               mode: str = 'same',
               boundary: str = 'constant') -> np.ndarray:
    """Apply 2D convolution to an image."""
    boundary_mode = {
        'constant': 'fill',
        'reflect': 'symm',
        'wrap': 'wrap'
    }.get(boundary, 'fill')

    result = signal.correlate2d(
        image.astype(np.float32),
        kernel.astype(np.float32),
        mode=mode,
        boundary=boundary_mode,
        fillvalue=0
    )

    return result.astype(np.float32)


def apply_kernel(image_matrix: ImageMatrix,
                 kernel_name: str,
                 normalize_output: bool = True) -> ImageMatrix:
    """Apply a named kernel to an ImageMatrix."""
    kernel = get_kernel_by_name(kernel_name)
    input_data = image_matrix.as_matrix()

    output = convolve2d(input_data, kernel)

    if normalize_output:
        if 'edge' in kernel_name.lower() or 'sobel' in kernel_name.lower() or 'laplacian' in kernel_name.lower():
            output = np.abs(output)

        min_val, max_val = output.min(), output.max()
        if max_val - min_val > 1e-8:
            output = (output - min_val) / (max_val - min_val)
        else:
            output = np.clip(output, 0, 1)

    result = ImageMatrix(output, f"{image_matrix.name}_{kernel_name}")
    result.history = image_matrix.history.copy()
    result.history.append(('convolution', kernel_name))
    return result
