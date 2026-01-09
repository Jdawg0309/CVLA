#!/usr/bin/env python3
"""
CVLA Vision Example: Image Processing as Linear Algebra

This script demonstrates how images are matrices and how convolutions
(the core operation of CNNs) work at the pixel level.

Run from CVLA root: python samples/vision_example.py
"""

import sys
import os

# Add CVLA root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from vision import (
    ImageMatrix, create_sample_image, load_image,
    SOBEL_X, SOBEL_Y, GAUSSIAN_BLUR, SHARPEN, LAPLACIAN,
    apply_kernel, convolve2d, list_kernels,
    AffineTransform, apply_affine_transform, normalize_image,
    ConvolutionVisualizer, compute_gradient_magnitude
)


def example_image_as_matrix():
    """
    Example 1: Images are matrices.

    Each pixel is a number. Grayscale images are 2D matrices.
    RGB images are 3D tensors (H x W x 3).
    """
    print("\n" + "=" * 50)
    print("Example 1: Images are Matrices")
    print("=" * 50)

    # Create an 8x8 checkerboard image
    img = create_sample_image(8, 'checkerboard')
    print(f"\nCreated: {img}")
    print(f"Shape: {img.shape} (8 rows x 8 columns)")
    print(f"Total pixels: {img.height * img.width} = 64")

    # View the raw matrix
    print("\nMatrix view (pixel values 0.0 to 1.0):")
    matrix = img.as_matrix()
    for row in matrix:
        print("  " + "".join("1 " if v > 0.5 else "0 " for v in row))

    # Statistics
    stats = img.get_statistics()
    print(f"\nStatistics:")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Std:  {stats['std']:.3f}")

    return img


def example_convolution_kernels():
    """
    Example 2: Convolution kernels detect features.

    A kernel is a small matrix that slides over the image.
    At each position, we compute a weighted sum.
    """
    print("\n" + "=" * 50)
    print("Example 2: Convolution Kernels")
    print("=" * 50)

    print("\nKernel: Sobel X (vertical edge detector)")
    print("This kernel highlights vertical edges:")
    print(SOBEL_X)

    print("\nKernel: Sobel Y (horizontal edge detector)")
    print("This kernel highlights horizontal edges:")
    print(SOBEL_Y)

    print("\nKernel: Gaussian Blur (smoothing)")
    print("This kernel averages neighbors (weighted):")
    print(GAUSSIAN_BLUR)

    print("\nAll available kernels:")
    for name, desc in list_kernels():
        print(f"  {name}: {desc}")


def example_apply_convolution(img):
    """
    Example 3: Applying convolution to an image.

    This is the core operation in CNNs!
    """
    print("\n" + "=" * 50)
    print("Example 3: Applying Convolution")
    print("=" * 50)

    # Apply Sobel X to detect vertical edges
    result = apply_kernel(img, 'sobel_x')
    print(f"\nApplied Sobel X to {img.name}")
    print(f"Result: {result}")

    # Show the before/after
    print("\nOriginal (8x8 checkerboard):")
    orig_matrix = img.as_matrix()
    for row in orig_matrix:
        print("  " + "".join("# " if v > 0.5 else ". " for v in row))

    print("\nAfter Sobel X (edge response):")
    result_matrix = result.as_matrix()
    for row in result_matrix:
        print("  " + "".join("# " if v > 0.3 else ". " for v in row))

    return result


def example_convolution_step_by_step():
    """
    Example 4: Understanding convolution step-by-step.

    Shows exactly what happens at one position.
    """
    print("\n" + "=" * 50)
    print("Example 4: Convolution Step-by-Step")
    print("=" * 50)

    # Create small image
    img = create_sample_image(5, 'edges')
    print(f"\nImage ({img.height}x{img.width}):")
    matrix = img.as_matrix()
    for row in matrix:
        print("  " + " ".join(f"{v:.1f}" for v in row))

    # Visualize at position (2, 2)
    viz = ConvolutionVisualizer(img, SOBEL_X)
    step = viz.get_kernel_position_info(2, 2)

    print(f"\nAt position (2, 2):")
    print(f"Image region (3x3 window):")
    for row in step['region']:
        print("  " + " ".join(f"{v:.1f}" for v in row))

    print(f"\nKernel:")
    for row in step['kernel']:
        print("  " + " ".join(f"{v:>4.0f}" for v in row))

    print(f"\nElement-wise products:")
    for row in step['products']:
        print("  " + " ".join(f"{v:>5.1f}" for v in row))

    print(f"\nSum of products (output value): {step['output']:.2f}")


def example_affine_transforms():
    """
    Example 5: Affine transformations are matrix operations.

    Rotation, scaling, translation can all be expressed as matrices.
    """
    print("\n" + "=" * 50)
    print("Example 5: Affine Transformations")
    print("=" * 50)

    # Create rotation matrix
    transform = AffineTransform()
    transform.rotate(30)  # 30 degrees

    print("\nRotation matrix (30 degrees):")
    print(transform.matrix)

    # Transform a point
    original = (1, 0)
    new_x, new_y = transform.transform_point(*original)
    print(f"\nPoint {original} rotated 30 degrees:")
    print(f"  New position: ({new_x:.3f}, {new_y:.3f})")

    # Compose transformations
    transform2 = AffineTransform()
    transform2.rotate(15).scale(2.0).translate(1, 1)
    print("\nComposed transformation (rotate 15 + scale 2x + translate):")
    print(transform2.matrix)


def example_ml_cv_pipeline():
    """
    Example 6: Complete ML/CV pipeline.

    Image -> Normalize -> Convolve -> Result
    """
    print("\n" + "=" * 50)
    print("Example 6: ML/CV Pipeline")
    print("=" * 50)

    # Step 1: Load/create image
    img = create_sample_image(16, 'circle')
    print(f"\n1. Input: {img}")

    # Step 2: Normalize (like preprocessing for neural networks)
    normalized = normalize_image(img)
    print(f"2. Normalized: mean={np.mean(normalized.data):.3f}, std={np.std(normalized.data):.3f}")

    # Step 3: Apply edge detection
    edges = apply_kernel(img, 'edge_detect')
    print(f"3. Edge detection applied: {edges}")

    # Step 4: Apply blur for smoothing
    smoothed = apply_kernel(img, 'gaussian_blur')
    print(f"4. Blur applied: {smoothed}")

    # Show transformation history
    print(f"\nOperation history: {edges.history}")


def main():
    print("=" * 60)
    print("  CVLA Vision Module: Image Processing as Linear Algebra")
    print("=" * 60)
    print("""
    This example demonstrates how images and convolutions work
    at the mathematical level - the foundation of CNNs and ML.
    """)

    # Run examples
    img = example_image_as_matrix()
    example_convolution_kernels()
    example_apply_convolution(img)
    example_convolution_step_by_step()
    example_affine_transforms()
    example_ml_cv_pipeline()

    print("\n" + "=" * 60)
    print("KEY INSIGHT: Every image operation is matrix math!")
    print("=" * 60)
    print("""
    In CNNs:
    - Images = matrices/tensors
    - Kernels = learned feature detectors
    - Convolution = sliding dot product
    - Deep learning = stacking many such operations

    The kernels shown here (Sobel, Gaussian) are similar
    to what neural networks learn automatically from data!
    """)


if __name__ == "__main__":
    main()
