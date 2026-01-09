"""
CVLA Vision Module - Image Processing for Linear Algebra Visualization

This module provides tools to demonstrate how machine learning and computer vision
work at the pixel and matrix level. It treats images as:
- 2D grayscale matrices (H x W)
- 3D RGB tensors (H x W x 3)

The goal is to make the relationship between pixels, matrices, linear transformations,
and convolutions visually and mathematically explicit.
"""

from .image import ImageMatrix, load_image, create_sample_image
from .kernels import (
    SOBEL_X, SOBEL_Y, LAPLACIAN, GAUSSIAN_BLUR, SHARPEN, EDGE_DETECT,
    IDENTITY, BOX_BLUR, EMBOSS, get_kernel_by_name, list_kernels
)
from .convolution import (
    convolve2d, apply_kernel, visualize_convolution_step,
    ConvolutionVisualizer, compute_gradient_magnitude, compute_gradient_direction
)
from .transforms import (
    apply_affine_transform, create_rotation_matrix, create_scale_matrix,
    create_translation_matrix, normalize_image, AffineTransform
)

__all__ = [
    # Image handling
    'ImageMatrix', 'load_image', 'create_sample_image',
    # Kernels
    'SOBEL_X', 'SOBEL_Y', 'LAPLACIAN', 'GAUSSIAN_BLUR', 'SHARPEN',
    'EDGE_DETECT', 'IDENTITY', 'BOX_BLUR', 'EMBOSS',
    'get_kernel_by_name', 'list_kernels',
    # Convolution
    'convolve2d', 'apply_kernel', 'visualize_convolution_step',
    'ConvolutionVisualizer', 'compute_gradient_magnitude', 'compute_gradient_direction',
    # Transforms
    'apply_affine_transform', 'create_rotation_matrix', 'create_scale_matrix',
    'create_translation_matrix', 'normalize_image', 'AffineTransform',
]
