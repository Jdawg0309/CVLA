"""Convolution utilities for image processing."""

from domain.images.convolution.convolution import (
    convolve2d, apply_kernel, visualize_convolution_step,
    ConvolutionVisualizer, compute_gradient_magnitude,
    compute_gradient_direction, multi_scale_convolution,
)

__all__ = [
    'convolve2d', 'apply_kernel', 'visualize_convolution_step',
    'ConvolutionVisualizer', 'compute_gradient_magnitude', 'compute_gradient_direction',
    'multi_scale_convolution',
]
