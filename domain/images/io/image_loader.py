"""
Image loading utilities.
"""

import numpy as np
from typing import Tuple, Optional, Union
from pathlib import Path

from domain.images.image_matrix import ImageMatrix
from domain.images.image_samples import create_sample_image

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: Pillow not installed. Install with: pip install Pillow")


def load_image(path: Union[str, Path],
               max_size: Optional[Tuple[int, int]] = None,
               grayscale: bool = False) -> Optional[ImageMatrix]:
    """
    Load an image from disk and convert to ImageMatrix.
    """
    if not PIL_AVAILABLE:
        print("Error: Pillow is required for image loading")
        return None

    try:
        path = Path(path)
        img = Image.open(path)

        if grayscale:
            img = img.convert('L')
        elif img.mode == 'RGBA':
            img = img.convert('RGB')
        elif img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')

        if max_size is not None:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

        data = np.array(img, dtype=np.float32)

        return ImageMatrix(data, name=path.stem)

    except Exception as e:
        print(f"Error loading image: {e}")
        return None
