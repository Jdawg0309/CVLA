"""
Convolution operations for image processing.
"""

from domain.images.convolution.convolution_core import convolve2d, apply_kernel
from domain.images.convolution.convolution_visuals import visualize_convolution_step, ConvolutionVisualizer
from domain.images.convolution.convolution_gradients import compute_gradient_magnitude, compute_gradient_direction
from domain.images.convolution.convolution_multiscale import multi_scale_convolution

__all__ = [
    'convolve2d',
    'apply_kernel',
    'visualize_convolution_step',
    'ConvolutionVisualizer',
    'compute_gradient_magnitude',
    'compute_gradient_direction',
    'multi_scale_convolution',
]
