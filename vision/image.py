"""
Image Loading and Matrix Conversion for CVLA

This module demonstrates how images are fundamentally matrices:
- A grayscale image is a 2D matrix where each element is a pixel intensity (0-255 or 0.0-1.0)
- An RGB image is a 3D tensor (H x W x 3) where the third dimension represents color channels

ML/CV Insight:
    In machine learning, images are always converted to numerical arrays (tensors).
    Neural networks process these matrices through series of linear transformations
    (matrix multiplications) and non-linear activations.
"""

import numpy as np
from typing import Tuple, Optional, Union
from pathlib import Path

# Try to import PIL, provide fallback if not available
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: Pillow not installed. Install with: pip install Pillow")


class ImageMatrix:
    """
    Represents an image as a mathematical matrix/tensor.

    This class wraps image data and provides methods to:
    - View the image as a matrix
    - Extract color channels
    - Convert between color spaces
    - Apply transformations

    ML/CV Insight:
        This is similar to how PyTorch/TensorFlow handle images internally.
        The key insight is that an image IS a matrix - pixel values ARE numbers.
    """

    def __init__(self, data: np.ndarray, name: str = "image"):
        """
        Initialize an ImageMatrix from a numpy array.

        Args:
            data: Image data as numpy array
                  - Shape (H, W) for grayscale
                  - Shape (H, W, 3) for RGB
                  - Shape (H, W, 4) for RGBA
            name: Optional name for the image

        ML/CV Insight:
            Images are stored in (Height, Width, Channels) format.
            This is the standard format for most deep learning frameworks.
        """
        self.data = np.array(data, dtype=np.float32)
        self.name = name
        self._original = self.data.copy()
        self.history = []  # Track transformations applied

        # Normalize to 0-1 range if needed
        if self.data.max() > 1.0:
            self.data = self.data / 255.0
            self._original = self._original / 255.0

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the image matrix (H, W) or (H, W, C)."""
        return self.data.shape

    @property
    def height(self) -> int:
        """Return image height (number of rows in the matrix)."""
        return self.data.shape[0]

    @property
    def width(self) -> int:
        """Return image width (number of columns in the matrix)."""
        return self.data.shape[1]

    @property
    def channels(self) -> int:
        """Return number of color channels (1 for grayscale, 3 for RGB, 4 for RGBA)."""
        if len(self.data.shape) == 2:
            return 1
        return self.data.shape[2]

    @property
    def is_grayscale(self) -> bool:
        """Check if the image is grayscale (single channel)."""
        return self.channels == 1

    @property
    def is_rgb(self) -> bool:
        """Check if the image is RGB (3 channels)."""
        return self.channels == 3

    def as_matrix(self) -> np.ndarray:
        """
        Return the image as a pure matrix representation.

        For grayscale: returns 2D matrix (H x W)
        For RGB: returns the luminance (weighted average of channels)

        ML/CV Insight:
            This is how grayscale conversion works in image processing.
            The weights [0.299, 0.587, 0.114] approximate human perception
            of brightness (we're more sensitive to green).
        """
        if self.is_grayscale:
            return self.data if len(self.data.shape) == 2 else self.data[:, :, 0]

        # Convert RGB to grayscale using luminance formula
        # Y = 0.299*R + 0.587*G + 0.114*B
        return (0.299 * self.data[:, :, 0] +
                0.587 * self.data[:, :, 1] +
                0.114 * self.data[:, :, 2])

    def get_channel(self, channel: Union[int, str]) -> np.ndarray:
        """
        Extract a single color channel as a 2D matrix.

        Args:
            channel: Channel index (0, 1, 2) or name ('R', 'G', 'B')

        Returns:
            2D numpy array representing the channel

        ML/CV Insight:
            Each channel is a separate 2D matrix. In CNNs, we often process
            channels independently before combining them.
        """
        if self.is_grayscale:
            return self.as_matrix()

        channel_map = {'R': 0, 'G': 1, 'B': 2, 'r': 0, 'g': 1, 'b': 2}
        if isinstance(channel, str):
            channel = channel_map.get(channel, 0)

        return self.data[:, :, channel]

    def get_rgb_planes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all three RGB channels as separate matrices.

        Returns:
            Tuple of (R, G, B) matrices

        ML/CV Insight:
            This separation is fundamental to understanding how CNNs
            process color images. Each plane is essentially a grayscale
            image representing that color's intensity.
        """
        if self.is_grayscale:
            gray = self.as_matrix()
            return gray, gray, gray

        return (
            self.data[:, :, 0],  # Red channel
            self.data[:, :, 1],  # Green channel
            self.data[:, :, 2]   # Blue channel
        )

    def to_grayscale(self) -> 'ImageMatrix':
        """
        Convert to grayscale and return new ImageMatrix.

        ML/CV Insight:
            Grayscale conversion reduces a 3D tensor to a 2D matrix,
            making it easier to visualize and understand operations.
        """
        gray_data = self.as_matrix()
        result = ImageMatrix(gray_data, f"{self.name}_gray")
        result.history = self.history.copy()
        result.history.append(('grayscale', None))
        return result

    def get_pixel_region(self, row: int, col: int, size: int = 3) -> np.ndarray:
        """
        Extract a small region around a pixel (useful for visualizing convolution).

        Args:
            row: Center row
            col: Center column
            size: Region size (must be odd)

        ML/CV Insight:
            This is exactly what a convolution kernel "sees" at each position.
            The kernel slides over the image, looking at small regions like this.
        """
        half = size // 2
        matrix = self.as_matrix()

        # Handle boundaries with padding
        padded = np.pad(matrix, half, mode='constant', constant_values=0)

        return padded[row:row + size, col:col + size]

    def reset(self):
        """Reset image to original state."""
        self.data = self._original.copy()
        self.history.clear()

    def apply_transform(self, matrix: np.ndarray, transform_name: str = "transform"):
        """
        Apply a transformation matrix to the image coordinates.

        ML/CV Insight:
            This demonstrates that geometric transformations (rotation, scaling)
            are just matrix multiplications on the coordinate system.
        """
        self.history.append((transform_name, matrix.copy()))

    def get_statistics(self) -> dict:
        """
        Return statistical information about the image matrix.

        ML/CV Insight:
            These statistics are often used for normalization in neural networks.
            Batch normalization uses mean and std to normalize activations.
        """
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


def load_image(path: Union[str, Path],
               max_size: Optional[Tuple[int, int]] = None,
               grayscale: bool = False) -> Optional[ImageMatrix]:
    """
    Load an image from disk and convert to ImageMatrix.

    Args:
        path: Path to image file (PNG, JPG, etc.)
        max_size: Optional (width, height) to resize large images
        grayscale: If True, convert to grayscale

    Returns:
        ImageMatrix object or None if loading fails

    ML/CV Insight:
        This is the first step in any image processing pipeline.
        Images are loaded from files and converted to numerical arrays.
    """
    if not PIL_AVAILABLE:
        print("Error: Pillow is required for image loading")
        return None

    try:
        path = Path(path)
        img = Image.open(path)

        # Convert mode if necessary
        if grayscale:
            img = img.convert('L')  # 'L' is grayscale mode
        elif img.mode == 'RGBA':
            img = img.convert('RGB')  # Remove alpha for simplicity
        elif img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')

        # Resize if max_size specified
        if max_size is not None:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Convert to numpy array
        data = np.array(img, dtype=np.float32)

        return ImageMatrix(data, name=path.stem)

    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def create_sample_image(size: int = 64, pattern: str = 'gradient') -> ImageMatrix:
    """
    Create a sample image for testing and demonstration.

    Args:
        size: Image size (creates size x size image)
        pattern: Type of pattern - 'gradient', 'checkerboard', 'circle', 'edges'

    Returns:
        ImageMatrix with the generated pattern

    ML/CV Insight:
        These synthetic images help visualize how different operations
        affect image data. Gradients show smooth transitions, checkerboards
        show edge detection, circles show shape detection.
    """
    if pattern == 'gradient':
        # Horizontal gradient - shows how pixel values change smoothly
        x = np.linspace(0, 1, size)
        data = np.tile(x, (size, 1))
        name = 'gradient'

    elif pattern == 'checkerboard':
        # Checkerboard - useful for testing edge detection
        block_size = max(1, size // 8)
        data = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if ((i // block_size) + (j // block_size)) % 2 == 0:
                    data[i, j] = 1.0
        name = 'checkerboard'

    elif pattern == 'circle':
        # Circle - useful for testing shape detection
        y, x = np.ogrid[:size, :size]
        center = size // 2
        radius = size // 3
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
        data = mask.astype(np.float32)
        name = 'circle'

    elif pattern == 'edges':
        # Sharp edges - useful for testing edge detection kernels
        data = np.zeros((size, size))
        # Vertical edge
        data[:, size // 2:] = 1.0
        # Horizontal edge
        data[size // 2:, :size // 4] = 0.5
        name = 'edges'

    elif pattern == 'noise':
        # Random noise - useful for testing blur/smoothing
        data = np.random.rand(size, size).astype(np.float32)
        name = 'noise'

    elif pattern == 'rgb_gradient':
        # RGB gradient - demonstrates color channels
        data = np.zeros((size, size, 3), dtype=np.float32)
        data[:, :, 0] = np.linspace(0, 1, size)  # Red gradient horizontal
        data[:, :, 1] = np.linspace(0, 1, size).reshape(-1, 1)  # Green gradient vertical
        data[:, :, 2] = 0.5  # Constant blue
        name = 'rgb_gradient'

    else:
        # Default: simple gradient
        x = np.linspace(0, 1, size)
        data = np.tile(x, (size, 1))
        name = 'default'

    return ImageMatrix(data.astype(np.float32), name=name)
