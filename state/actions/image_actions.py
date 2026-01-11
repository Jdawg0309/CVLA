"""
Image-related action definitions.
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass(frozen=True)
class LoadImage:
    """Load an image from file path."""
    path: str
    max_size: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class CreateSampleImage:
    """Create a sample/test image."""
    pattern: str  # 'gradient', 'checkerboard', 'circle', 'edges', 'noise'
    size: int


@dataclass(frozen=True)
class ApplyKernel:
    """Apply a convolution kernel to the current image."""
    kernel_name: str


@dataclass(frozen=True)
class ApplyTransform:
    """Apply an affine transform to the current image."""
    rotation: float  # degrees
    scale: float


@dataclass(frozen=True)
class FlipImageHorizontal:
    """Flip the current image horizontally."""
    pass


@dataclass(frozen=True)
class UseResultAsInput:
    """Use the processed image as the new input."""
    pass


@dataclass(frozen=True)
class ClearImage:
    """Clear both current and processed images."""
    pass
