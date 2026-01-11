"""
Convolution operations for image processing.
"""

from vision.convolution_core import convolve2d, apply_kernel
from vision.convolution_visuals import visualize_convolution_step, ConvolutionVisualizer
from vision.convolution_gradients import compute_gradient_magnitude, compute_gradient_direction
from vision.convolution_multiscale import multi_scale_convolution

__all__ = [
    'convolve2d',
    'apply_kernel',
    'visualize_convolution_step',
    'ConvolutionVisualizer',
    'compute_gradient_magnitude',
    'compute_gradient_direction',
    'multi_scale_convolution',
]
