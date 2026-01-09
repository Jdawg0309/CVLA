"""
Convolution Operations for Image Processing

This module implements 2D convolution - the fundamental operation in CNNs.

ML/CV Insight:
    Convolution is the core operation in Convolutional Neural Networks.
    It works by sliding a kernel (small matrix) over the image and computing
    weighted sums at each position. This extracts local features like edges,
    textures, and patterns.

Mathematical Definition:
    For image I and kernel K of size (2m+1) x (2n+1):
    output[i,j] = sum over u,v of: I[i+u, j+v] * K[u, v]

    This is technically "correlation" not convolution (kernel not flipped),
    but in deep learning we commonly call it convolution.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from scipy import signal

from .image import ImageMatrix
from .kernels import get_kernel_by_name, IDENTITY


def convolve2d(image: np.ndarray,
               kernel: np.ndarray,
               mode: str = 'same',
               boundary: str = 'constant') -> np.ndarray:
    """
    Apply 2D convolution to an image.

    Args:
        image: 2D input image array
        kernel: 2D convolution kernel
        mode: Output size mode
            - 'same': Output same size as input (default)
            - 'valid': Only positions where kernel fully overlaps
            - 'full': Full convolution output
        boundary: How to handle boundaries
            - 'constant': Pad with zeros (default)
            - 'reflect': Mirror image at boundaries
            - 'wrap': Wrap around (periodic)

    Returns:
        Convolved image

    ML/CV Insight:
        In CNNs, we almost always use 'same' mode with zero padding.
        This preserves spatial dimensions through the network.
        The boundary handling affects edge pixels - zeros can create artifacts.
    """
    # Use scipy for efficient convolution
    # Note: scipy's correlate2d is what we want (doesn't flip kernel)
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
    """
    Apply a named kernel to an ImageMatrix.

    Args:
        image_matrix: Input ImageMatrix
        kernel_name: Name of kernel from registry (e.g., 'sobel_x', 'gaussian_blur')
        normalize_output: If True, normalize result to 0-1 range

    Returns:
        New ImageMatrix with kernel applied

    ML/CV Insight:
        This is one "layer" of a CNN. The kernel extracts specific features.
        Multiple kernels applied in parallel create feature maps.
    """
    kernel = get_kernel_by_name(kernel_name)
    input_data = image_matrix.as_matrix()

    # Apply convolution
    output = convolve2d(input_data, kernel)

    # Normalize if requested
    if normalize_output:
        # For edge detection kernels, take absolute value
        if 'edge' in kernel_name.lower() or 'sobel' in kernel_name.lower() or 'laplacian' in kernel_name.lower():
            output = np.abs(output)

        # Normalize to 0-1
        min_val, max_val = output.min(), output.max()
        if max_val - min_val > 1e-8:
            output = (output - min_val) / (max_val - min_val)
        else:
            output = np.clip(output, 0, 1)

    result = ImageMatrix(output, f"{image_matrix.name}_{kernel_name}")
    result.history = image_matrix.history.copy()
    result.history.append(('convolution', kernel_name))
    return result


def visualize_convolution_step(image: np.ndarray,
                               kernel: np.ndarray,
                               position: Tuple[int, int],
                               padding: int = 0) -> Dict[str, Any]:
    """
    Visualize a single convolution step at a specific position.

    This shows exactly what happens when the kernel is applied to one position.

    Args:
        image: Input image (2D array)
        kernel: Convolution kernel
        position: (row, col) position in the image
        padding: Amount of zero padding applied

    Returns:
        Dictionary containing:
        - 'region': The image region under the kernel
        - 'kernel': The kernel values
        - 'products': Element-wise products
        - 'output': Final summed value
        - 'position': Where in the output this goes

    ML/CV Insight:
        This step-by-step view shows exactly how CNNs compute each output pixel.
        The neural network learns the kernel weights through backpropagation.
    """
    row, col = position
    k_height, k_width = kernel.shape
    pad_h, pad_w = k_height // 2, k_width // 2

    # Pad the image
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)),
                    mode='constant', constant_values=0)

    # Extract region
    region = padded[row:row + k_height, col:col + k_width]

    # Compute products
    products = region * kernel

    # Sum to get output
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

    This class helps demonstrate how convolution works step-by-step,
    which is essential for understanding CNNs.

    ML/CV Insight:
        Understanding convolution visually is key to understanding deep learning.
        Each position shows: input region -> kernel weights -> output value
    """

    def __init__(self, image: ImageMatrix, kernel: np.ndarray):
        """
        Initialize the visualizer.

        Args:
            image: Input ImageMatrix
            kernel: Convolution kernel to apply
        """
        self.image = image.as_matrix()
        self.kernel = kernel.astype(np.float32)
        self.output = None
        self.current_position = (0, 0)
        self.step_data = None

        # Compute full output
        self._compute_full_output()

    def _compute_full_output(self):
        """Compute the full convolution output."""
        self.output = convolve2d(self.image, self.kernel)

    def get_kernel_position_info(self, row: int, col: int) -> Dict[str, Any]:
        """
        Get detailed info about convolution at a specific position.

        Args:
            row: Row position
            col: Column position

        Returns:
            Dictionary with visualization data
        """
        self.current_position = (row, col)
        self.step_data = visualize_convolution_step(
            self.image, self.kernel, (row, col)
        )
        return self.step_data

    def get_sliding_animation_frames(self,
                                     max_frames: int = 100) -> List[Dict[str, Any]]:
        """
        Generate frames for animating the kernel sliding over the image.

        Args:
            max_frames: Maximum number of frames to generate

        Returns:
            List of frame data dictionaries

        ML/CV Insight:
            This animation shows how convolution is inherently parallel -
            each position is independent and could be computed simultaneously.
            GPUs exploit this parallelism for fast CNN inference.
        """
        h, w = self.image.shape
        k_h, k_w = self.kernel.shape

        # Calculate step size to limit frames
        total_positions = h * w
        step = max(1, total_positions // max_frames)

        frames = []
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
        """
        Get statistics about the convolution output (feature map).

        Returns:
            Dictionary with statistics

        ML/CV Insight:
            These statistics are similar to what batch normalization tracks.
            In CNNs, we normalize feature maps to improve training stability.
        """
        if self.output is None:
            self._compute_full_output()

        return {
            'mean': float(np.mean(self.output)),
            'std': float(np.std(self.output)),
            'min': float(np.min(self.output)),
            'max': float(np.max(self.output)),
            'sparsity': float((np.abs(self.output) < 0.01).mean()),  # % near zero
            'activation_ratio': float((self.output > 0).mean()),  # % positive
            'shape': self.output.shape
        }

    def compare_kernels(self, kernel_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Apply multiple kernels and compare results.

        Args:
            kernel_names: List of kernel names to apply

        Returns:
            Dictionary mapping kernel name to output array

        ML/CV Insight:
            In CNNs, multiple kernels are applied in parallel at each layer.
            This creates multiple "feature maps" that capture different aspects
            of the input (edges, textures, colors, etc.).
        """
        results = {}
        for name in kernel_names:
            kernel = get_kernel_by_name(name)
            output = convolve2d(self.image, kernel)

            # Normalize for comparison
            abs_output = np.abs(output)
            if abs_output.max() > 0:
                normalized = abs_output / abs_output.max()
            else:
                normalized = abs_output

            results[name] = normalized

        return results


def compute_gradient_magnitude(image: ImageMatrix) -> np.ndarray:
    """
    Compute gradient magnitude using Sobel operators.

    Args:
        image: Input ImageMatrix

    Returns:
        Gradient magnitude image

    ML/CV Insight:
        Gradient magnitude combines horizontal and vertical edge responses.
        This is often used as a pre-processing step or feature.
        magnitude = sqrt(Gx^2 + Gy^2)
    """
    from .kernels import SOBEL_X, SOBEL_Y

    data = image.as_matrix()
    gx = convolve2d(data, SOBEL_X)
    gy = convolve2d(data, SOBEL_Y)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # Normalize
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()

    return magnitude


def compute_gradient_direction(image: ImageMatrix) -> np.ndarray:
    """
    Compute gradient direction using Sobel operators.

    Args:
        image: Input ImageMatrix

    Returns:
        Gradient direction in radians (-pi to pi)

    ML/CV Insight:
        Gradient direction is used in algorithms like HOG (Histogram of
        Oriented Gradients) which is a classic feature descriptor for
        object detection.
    """
    from .kernels import SOBEL_X, SOBEL_Y

    data = image.as_matrix()
    gx = convolve2d(data, SOBEL_X)
    gy = convolve2d(data, SOBEL_Y)

    return np.arctan2(gy, gx)


def multi_scale_convolution(image: ImageMatrix,
                            kernel_name: str,
                            scales: List[int] = [3, 5, 7]) -> Dict[int, np.ndarray]:
    """
    Apply convolution at multiple scales.

    Args:
        image: Input ImageMatrix
        kernel_name: Base kernel to use
        scales: List of kernel sizes

    Returns:
        Dictionary mapping scale to output

    ML/CV Insight:
        Multi-scale analysis captures features at different sizes.
        This is similar to how Inception networks use multiple kernel sizes
        in parallel to capture both fine and coarse features.
    """
    from .kernels import create_gaussian_kernel

    results = {}
    data = image.as_matrix()

    for scale in scales:
        if kernel_name == 'gaussian_blur':
            # Create scaled Gaussian
            kernel = create_gaussian_kernel(scale, sigma=scale / 3)
        else:
            # Use base kernel (don't scale arbitrary kernels)
            kernel = get_kernel_by_name(kernel_name)

        output = convolve2d(data, kernel)
        results[scale] = output

    return results
