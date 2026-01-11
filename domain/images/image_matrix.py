"""
Image matrix representation and utilities.
"""

import numpy as np
from typing import Tuple, Optional, Union


class ImageMatrix:
    """
    Represents an image as a mathematical matrix/tensor.
    """

    def __init__(self, data: np.ndarray, name: str = "image"):
        self.data = np.array(data, dtype=np.float32)
        self.name = name
        self._original = self.data.copy()
        self.history = []

        if self.data.max() > 1.0:
            self.data = self.data / 255.0
            self._original = self._original / 255.0

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def channels(self) -> int:
        if len(self.data.shape) == 2:
            return 1
        return self.data.shape[2]

    @property
    def is_grayscale(self) -> bool:
        return self.channels == 1

    @property
    def is_rgb(self) -> bool:
        return self.channels == 3

    def as_matrix(self) -> np.ndarray:
        if self.is_grayscale:
            return self.data if len(self.data.shape) == 2 else self.data[:, :, 0]
        return (0.299 * self.data[:, :, 0] +
                0.587 * self.data[:, :, 1] +
                0.114 * self.data[:, :, 2])

    def get_channel(self, channel: Union[int, str]) -> np.ndarray:
        if self.is_grayscale:
            return self.as_matrix()

        channel_map = {'R': 0, 'G': 1, 'B': 2, 'r': 0, 'g': 1, 'b': 2}
        if isinstance(channel, str):
            channel = channel_map.get(channel, 0)

        return self.data[:, :, channel]

    def get_rgb_planes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.is_grayscale:
            gray = self.as_matrix()
            return gray, gray, gray

        return (
            self.data[:, :, 0],
            self.data[:, :, 1],
            self.data[:, :, 2]
        )

    def to_grayscale(self) -> 'ImageMatrix':
        gray_data = self.as_matrix()
        result = ImageMatrix(gray_data, f"{self.name}_gray")
        result.history = self.history.copy()
        result.history.append(('grayscale', None))
        return result

    def get_pixel_region(self, row: int, col: int, size: int = 3) -> np.ndarray:
        half = size // 2
        matrix = self.as_matrix()
        padded = np.pad(matrix, half, mode='constant', constant_values=0)
        return padded[row:row + size, col:col + size]

    def reset(self):
        self.data = self._original.copy()
        self.history.clear()

    def apply_transform(self, matrix: np.ndarray, transform_name: str = "transform"):
        self.history.append((transform_name, matrix.copy()))

    def get_statistics(self) -> dict:
        matrix = self.as_matrix()
        return {
            'mean': float(np.mean(matrix)),
            'std': float(np.std(matrix)),
            'min': float(np.min(matrix)),
            'max': float(np.max(matrix)),
            'shape': self.shape,
            'total_pixels': matrix.size
        }

    def __str__(self) -> str:
        shape_str = f"{self.height}x{self.width}"
        if not self.is_grayscale:
            shape_str += f"x{self.channels}"
        return f"ImageMatrix('{self.name}': {shape_str})"

    def __repr__(self) -> str:
        return self.__str__()
