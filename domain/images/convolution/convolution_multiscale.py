"""
Multi-scale convolution helpers.
"""

from typing import List, Dict

from domain.images.convolution.convolution_core import convolve2d
from domain.images.image import ImageMatrix
from domain.images.kernels.kernels import get_kernel_by_name, create_gaussian_kernel


def multi_scale_convolution(image: ImageMatrix,
                            kernel_name: str,
                            scales: List[int] = [3, 5, 7]) -> Dict[int, 'np.ndarray']:
    """Apply convolution at multiple scales."""
    results = {}
    data = image.as_matrix()

    for scale in scales:
        if kernel_name == 'gaussian_blur':
            kernel = create_gaussian_kernel(scale, sigma=scale / 3)
        else:
            kernel = get_kernel_by_name(kernel_name)

        output = convolve2d(data, kernel)
        results[scale] = output

    return results
