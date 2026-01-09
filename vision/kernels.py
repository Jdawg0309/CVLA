"""
Convolution Kernels for Image Processing

This module defines common convolution kernels (also called filters or masks).
A kernel is a small matrix that slides over an image to compute a weighted sum
at each position.

ML/CV Insight:
    In Convolutional Neural Networks (CNNs), kernels are LEARNED from data.
    The network discovers what kernels are useful for recognizing patterns.
    These predefined kernels show what kinds of features networks might learn:
    - Edge detectors (Sobel, Laplacian)
    - Smoothing/blur filters (Gaussian, Box)
    - Feature enhancers (Sharpen, Emboss)

Mathematical Foundation:
    Each kernel is a matrix of weights. When applied to an image region,
    we compute: output = sum(kernel * region)
    This is the discrete convolution operation.
"""

import numpy as np
from typing import Dict, List, Tuple


# =============================================================================
# EDGE DETECTION KERNELS
# =============================================================================

SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)
"""
Sobel X (Horizontal Edge Detector)

ML/CV Insight:
    Detects vertical edges (changes in the horizontal direction).
    The negative values on the left and positive on the right create
    a response when there's a brightness transition left-to-right.

    This is similar to what early layers of CNNs learn automatically.
"""

SOBEL_Y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)
"""
Sobel Y (Vertical Edge Detector)

ML/CV Insight:
    Detects horizontal edges (changes in the vertical direction).
    Combined with Sobel X, these can find edges in all orientations.

    Gradient magnitude = sqrt(Sobel_X^2 + Sobel_Y^2)
"""

LAPLACIAN = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=np.float32)
"""
Laplacian Edge Detector

ML/CV Insight:
    Detects edges in all directions at once (second derivative).
    The center pixel is compared to ALL its neighbors.
    Zero-crossings in the output indicate edges.

    More sensitive to noise than Sobel, but catches all edges.
"""

EDGE_DETECT = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
], dtype=np.float32)
"""
Simple Edge Detection (8-connected Laplacian variant)

ML/CV Insight:
    This kernel sums all neighbors (negative) and compares to center (positive).
    If a pixel matches its neighbors, output is ~0.
    If it differs significantly, output is high.
"""


# =============================================================================
# BLUR / SMOOTHING KERNELS
# =============================================================================

BOX_BLUR = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=np.float32) / 9.0
"""
Box Blur (Mean Filter)

ML/CV Insight:
    Replaces each pixel with the average of its 3x3 neighborhood.
    Simple but can create blocky artifacts on edges.

    In neural networks, average pooling is similar but samples rather than blurs.
"""

GAUSSIAN_BLUR = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32) / 16.0
"""
Gaussian Blur (3x3 approximation)

ML/CV Insight:
    Weighted average that gives more importance to center pixels.
    Produces smoother results than box blur.

    Pre-processing images with Gaussian blur helps reduce noise
    before feeding to neural networks.
"""

GAUSSIAN_BLUR_5x5 = np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
], dtype=np.float32) / 256.0
"""
Gaussian Blur (5x5)

ML/CV Insight:
    Larger kernel = stronger blur effect.
    The weights approximate a 2D Gaussian distribution.
"""


# =============================================================================
# SHARPENING / ENHANCEMENT KERNELS
# =============================================================================

SHARPEN = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)
"""
Sharpen Filter

ML/CV Insight:
    Enhances edges by amplifying the center pixel and subtracting neighbors.
    This is like adding the Laplacian to the original image:
    sharpened = original + (original - blurred)

    Similar to what happens in residual connections in ResNets!
"""

EMBOSS = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
], dtype=np.float32)
"""
Emboss Filter

ML/CV Insight:
    Creates a 3D embossed effect by detecting edges at an angle.
    The asymmetric weights create directional shading.
"""


# =============================================================================
# IDENTITY / UTILITY KERNELS
# =============================================================================

IDENTITY = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
], dtype=np.float32)
"""
Identity Kernel

ML/CV Insight:
    Outputs the input unchanged. Useful as a baseline or for testing.
    In neural networks, identity mappings (skip connections) help
    gradient flow in deep networks.
"""


# =============================================================================
# SPECIALIZED KERNELS
# =============================================================================

PREWITT_X = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)
"""
Prewitt X (Simpler edge detector)

ML/CV Insight:
    Similar to Sobel but with equal weights. Slightly less noise-resistant.
"""

PREWITT_Y = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
], dtype=np.float32)
"""
Prewitt Y
"""

RIDGE_DETECT = np.array([
    [-1, -1, -1],
    [2, 2, 2],
    [-1, -1, -1]
], dtype=np.float32)
"""
Horizontal Ridge Detector

ML/CV Insight:
    Detects horizontal lines/ridges in the image.
    Useful for detecting elongated structures.
"""


# =============================================================================
# KERNEL REGISTRY AND UTILITIES
# =============================================================================

KERNEL_REGISTRY: Dict[str, np.ndarray] = {
    'sobel_x': SOBEL_X,
    'sobel_y': SOBEL_Y,
    'laplacian': LAPLACIAN,
    'edge_detect': EDGE_DETECT,
    'box_blur': BOX_BLUR,
    'gaussian_blur': GAUSSIAN_BLUR,
    'gaussian_blur_5x5': GAUSSIAN_BLUR_5x5,
    'sharpen': SHARPEN,
    'emboss': EMBOSS,
    'identity': IDENTITY,
    'prewitt_x': PREWITT_X,
    'prewitt_y': PREWITT_Y,
    'ridge_detect': RIDGE_DETECT,
}

KERNEL_DESCRIPTIONS: Dict[str, str] = {
    'sobel_x': 'Detect vertical edges (horizontal gradient)',
    'sobel_y': 'Detect horizontal edges (vertical gradient)',
    'laplacian': 'Detect edges in all directions',
    'edge_detect': 'Strong edge detection (8-connected)',
    'box_blur': 'Simple averaging blur',
    'gaussian_blur': 'Weighted blur (3x3)',
    'gaussian_blur_5x5': 'Stronger blur (5x5)',
    'sharpen': 'Enhance edges and details',
    'emboss': 'Create 3D embossed effect',
    'identity': 'No change (pass-through)',
    'prewitt_x': 'Simpler vertical edge detector',
    'prewitt_y': 'Simpler horizontal edge detector',
    'ridge_detect': 'Detect horizontal lines/ridges',
}


def get_kernel_by_name(name: str) -> np.ndarray:
    """
    Get a kernel by its name.

    Args:
        name: Kernel name (e.g., 'sobel_x', 'gaussian_blur')

    Returns:
        Kernel as numpy array, or identity kernel if name not found

    ML/CV Insight:
        This mimics how neural networks select which learned filters to apply.
    """
    return KERNEL_REGISTRY.get(name.lower(), IDENTITY).copy()


def list_kernels() -> List[Tuple[str, str]]:
    """
    List all available kernels with descriptions.

    Returns:
        List of (name, description) tuples
    """
    return [(name, KERNEL_DESCRIPTIONS.get(name, ''))
            for name in KERNEL_REGISTRY.keys()]


def create_gaussian_kernel(size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """
    Create a Gaussian kernel of arbitrary size.

    Args:
        size: Kernel size (must be odd)
        sigma: Standard deviation of the Gaussian

    Returns:
        Normalized Gaussian kernel

    ML/CV Insight:
        Gaussian kernels are parameterized by sigma. Larger sigma = more blur.
        This is similar to how neural network layers have learnable parameters.
    """
    if size % 2 == 0:
        size += 1  # Ensure odd size

    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)

    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # Normalize so sum = 1
    return kernel / kernel.sum()


def kernel_to_string(kernel: np.ndarray, precision: int = 2) -> str:
    """
    Convert a kernel to a formatted string for display.

    Args:
        kernel: The kernel matrix
        precision: Decimal places to show

    Returns:
        Formatted string representation
    """
    rows = []
    for row in kernel:
        row_str = [f"{v:>{precision + 4}.{precision}f}" for v in row]
        rows.append("[" + ", ".join(row_str) + "]")
    return "[\n  " + ",\n  ".join(rows) + "\n]"


def visualize_kernel_weights(kernel: np.ndarray) -> str:
    """
    Create ASCII visualization of kernel weights.

    ML/CV Insight:
        Visualizing kernel weights helps understand what features they detect.
        Positive weights (shown as +/O) enhance matching patterns.
        Negative weights (shown as -/o) suppress mismatches.
    """
    # Normalize to -1 to 1 range
    max_abs = max(abs(kernel.min()), abs(kernel.max()))
    if max_abs > 0:
        normalized = kernel / max_abs
    else:
        normalized = kernel

    symbols = []
    for row in normalized:
        row_symbols = []
        for val in row:
            if val > 0.5:
                row_symbols.append('O')
            elif val > 0.1:
                row_symbols.append('+')
            elif val < -0.5:
                row_symbols.append('o')
            elif val < -0.1:
                row_symbols.append('-')
            else:
                row_symbols.append('.')
        symbols.append(' '.join(row_symbols))

    return '\n'.join(symbols)
