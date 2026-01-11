"""
Optional vision imports for sidebar image operations.
"""

try:
    from domain.images import (
        ImageMatrix, load_image, create_sample_image,
        list_kernels, get_kernel_by_name, apply_kernel,
        ConvolutionVisualizer, AffineTransform, apply_affine_transform,
    )
    VISION_AVAILABLE = True
    VISION_IMPORT_ERROR = None
except Exception as exc:
    VISION_AVAILABLE = False
    VISION_IMPORT_ERROR = str(exc)
    ImageMatrix = None
    load_image = None
    create_sample_image = None
    list_kernels = None
    get_kernel_by_name = None
    apply_kernel = None
    ConvolutionVisualizer = None
    AffineTransform = None
    apply_affine_transform = None
