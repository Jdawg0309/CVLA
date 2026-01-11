"""
Sample image generation utilities.
"""

import numpy as np

from domain.images.image_matrix import ImageMatrix


def create_sample_image(size: int = 64, pattern: str = 'gradient') -> ImageMatrix:
    """
    Create a sample image for testing and demonstration.
    """
    if pattern == 'gradient':
        x = np.linspace(0, 1, size)
        data = np.tile(x, (size, 1))
        name = 'gradient'

    elif pattern == 'checkerboard':
        block_size = max(1, size // 8)
        data = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if ((i // block_size) + (j // block_size)) % 2 == 0:
                    data[i, j] = 1.0
        name = 'checkerboard'

    elif pattern == 'circle':
        y, x = np.ogrid[:size, :size]
        center = size // 2
        radius = size // 3
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
        data = mask.astype(np.float32)
        name = 'circle'

    elif pattern == 'edges':
        data = np.zeros((size, size))
        data[:, size // 2:] = 1.0
        data[size // 2:, :size // 4] = 0.5
        name = 'edges'

    elif pattern == 'noise':
        data = np.random.rand(size, size).astype(np.float32)
        name = 'noise'

    elif pattern == 'rgb_gradient':
        data = np.zeros((size, size, 3), dtype=np.float32)
        data[:, :, 0] = np.linspace(0, 1, size)
        data[:, :, 1] = np.linspace(0, 1, size).reshape(-1, 1)
        data[:, :, 2] = 0.5
        name = 'rgb_gradient'

    else:
        x = np.linspace(0, 1, size)
        data = np.tile(x, (size, 1))
        name = 'default'

    return ImageMatrix(data.astype(np.float32), name=name)
