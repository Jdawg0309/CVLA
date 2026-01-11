"""
Convolution visualization helpers.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any

from vision.convolution_core import convolve2d
from vision.image import ImageMatrix
from vision.kernels import get_kernel_by_name


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
        self.output = None
        self.current_position = (0, 0)
        self.step_data = None
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
        results = {}
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
