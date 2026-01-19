"""
Convolution operations for image processing.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal

from domain.images.image import ImageMatrix
from domain.images.kernels.kernels import (
    SOBEL_X,
    SOBEL_Y,
    get_kernel_by_name,
    create_gaussian_kernel,
)


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


def compute_gradient_magnitude(image: ImageMatrix) -> np.ndarray:
    """Compute gradient magnitude using Sobel operators."""
    data = image.as_matrix()
    gx = convolve2d(data, SOBEL_X)
    gy = convolve2d(data, SOBEL_Y)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()

    return magnitude


def compute_gradient_direction(image: ImageMatrix) -> np.ndarray:
    """Compute gradient direction using Sobel operators."""
    data = image.as_matrix()
    gx = convolve2d(data, SOBEL_X)
    gy = convolve2d(data, SOBEL_Y)

    return np.arctan2(gy, gx)


def multi_scale_convolution(image: ImageMatrix,
                            kernel_name: str,
                            scales: List[int] = [3, 5, 7]) -> Dict[int, np.ndarray]:
    """Apply convolution at multiple scales."""
    results: Dict[int, np.ndarray] = {}
    data = image.as_matrix()

    for scale in scales:
        if kernel_name == 'gaussian_blur':
            kernel = create_gaussian_kernel(scale, sigma=scale / 3)
        else:
            kernel = get_kernel_by_name(kernel_name)

        output = convolve2d(data, kernel)
        results[scale] = output

    return results


def visualize_convolution_step(image: np.ndarray,
                               kernel: np.ndarray,
                               position: Tuple[int, int],
                               padding: int = 0) -> Dict[str, Any]:
    """Visualize a single convolution step at a specific position."""
    row, col = position
    k_height, k_width = kernel.shape
    pad_h, pad_w = k_height // 2, k_width // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)),
                    mode='constant', constant_values=0)
    region = padded[row:row + k_height, col:col + k_width]
    products = region * kernel
    output_value = products.sum()

    return {
        'region': region.copy(),
        'kernel': kernel.copy(),
        'products': products.copy(),
        'output': float(output_value),
        'position': position,
        'formula': f"sum({region.flatten()} * {kernel.flatten()}) = {output_value:.4f}"
    }


class ConvolutionVisualizer:
    """
    Interactive visualizer for convolution operations.
    """

    def __init__(self, image: ImageMatrix, kernel: np.ndarray):
        self.image = image.as_matrix()
        self.kernel = kernel.astype(np.float32)
        self.output: Optional[np.ndarray] = None
        self.current_position = (0, 0)
        self.step_data: Optional[Dict[str, Any]] = None
        self._compute_full_output()

    def _compute_full_output(self):
        """Compute the full convolution output."""
        self.output = convolve2d(self.image, self.kernel)

    def get_kernel_position_info(self, row: int, col: int) -> Dict[str, Any]:
        """Get detailed info about convolution at a specific position."""
        self.current_position = (row, col)
        self.step_data = visualize_convolution_step(
            self.image, self.kernel, (row, col)
        )
        return self.step_data

    def get_sliding_animation_frames(self,
                                     max_frames: int = 100) -> List[Dict[str, Any]]:
        """Generate frames for animating the kernel sliding over the image."""
        h, w = self.image.shape
        total_positions = h * w
        step = max(1, total_positions // max_frames)

        frames: List[Dict[str, Any]] = []
        frame_count = 0

        for i in range(h):
            for j in range(w):
                if frame_count % step == 0:
                    frames.append(self.get_kernel_position_info(i, j))
                frame_count += 1

                if len(frames) >= max_frames:
                    return frames

        return frames

    def get_feature_map_statistics(self) -> Dict[str, Any]:
        """Get statistics about the convolution output (feature map)."""
        if self.output is None:
            self._compute_full_output()

        return {
            'mean': float(np.mean(self.output)),
            'std': float(np.std(self.output)),
            'min': float(np.min(self.output)),
            'max': float(np.max(self.output)),
            'sparsity': float((np.abs(self.output) < 0.01).mean()),
            'activation_ratio': float((self.output > 0).mean()),
            'shape': self.output.shape
        }

    def compare_kernels(self, kernel_names: List[str]) -> Dict[str, np.ndarray]:
        """Apply multiple kernels and compare results."""
        results: Dict[str, np.ndarray] = {}
        for name in kernel_names:
            kernel = get_kernel_by_name(name)
            output = convolve2d(self.image, kernel)

            abs_output = np.abs(output)
            if abs_output.max() > 0:
                normalized = abs_output / abs_output.max()
            else:
                normalized = abs_output

            results[name] = normalized

        return results


__all__ = [
    'convolve2d',
    'apply_kernel',
    'visualize_convolution_step',
    'ConvolutionVisualizer',
    'compute_gradient_magnitude',
    'compute_gradient_direction',
    'multi_scale_convolution',
]
