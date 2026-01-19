"""
Convolution kernel registry and utilities.
"""

import numpy as np
from typing import Dict, List, Tuple

SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

SOBEL_Y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)

LAPLACIAN = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=np.float32)

EDGE_DETECT = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
], dtype=np.float32)

PREWITT_X = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

PREWITT_Y = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
], dtype=np.float32)

RIDGE_DETECT = np.array([
    [-1, -1, -1],
    [2, 2, 2],
    [-1, -1, -1]
], dtype=np.float32)

BOX_BLUR = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=np.float32) / 9.0

GAUSSIAN_BLUR = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32) / 16.0

GAUSSIAN_BLUR_5x5 = np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
], dtype=np.float32) / 256.0

SHARPEN = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

EMBOSS = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
], dtype=np.float32)

IDENTITY = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
], dtype=np.float32)

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
    """Get a kernel by its name."""
    return KERNEL_REGISTRY.get(name.lower(), IDENTITY).copy()


def list_kernels() -> List[Tuple[str, str]]:
    """List all available kernels with descriptions."""
    return [(kernel, KERNEL_DESCRIPTIONS.get(kernel, ''))
            for kernel in KERNEL_REGISTRY.keys()]


def create_gaussian_kernel(size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """Create a Gaussian kernel of arbitrary size."""
    if size % 2 == 0:
        size += 1

    center = size // 2
    kernel = np.zeros((size, size), dtype=np.float32)

    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    return kernel / kernel.sum()


def kernel_to_string(kernel: np.ndarray, precision: int = 2) -> str:
    """Convert a kernel to a formatted string for display."""
    rows = []
    for row in kernel:
        row_str = [f"{v:>{precision + 4}.{precision}f}" for v in row]
        rows.append("[" + ", ".join(row_str) + "]")
    return "[\n  " + ",\n  ".join(rows) + "\n]"


def visualize_kernel_weights(kernel: np.ndarray) -> str:
    """Create ASCII visualization of kernel weights."""
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
