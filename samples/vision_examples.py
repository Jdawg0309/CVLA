"""
Vision example functions.
"""

import numpy as np
from vision import (
    ImageMatrix, create_sample_image, load_image,
    SOBEL_X, SOBEL_Y, GAUSSIAN_BLUR, SHARPEN, LAPLACIAN,
    apply_kernel, convolve2d, list_kernels,
    AffineTransform, apply_affine_transform, normalize_image,
    ConvolutionVisualizer, compute_gradient_magnitude
)


def example_image_as_matrix():
    print("\n" + "=" * 50)
    print("Example 1: Images are Matrices")
    print("=" * 50)

    img = create_sample_image(8, 'checkerboard')
    print(f"\nCreated: {img}")
    print(f"Shape: {img.shape} (8 rows x 8 columns)")
    print(f"Total pixels: {img.height * img.width} = 64")

    print("\nMatrix view (pixel values 0.0 to 1.0):")
    matrix = img.as_matrix()
    for row in matrix:
        print("  " + "".join("1 " if v > 0.5 else "0 " for v in row))

    stats = img.get_statistics()
    print(f"\nStatistics:")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Std:  {stats['std']:.3f}")

    return img


def example_convolution_kernels():
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
    print("\n" + "=" * 50)
    print("Example 3: Applying Convolution")
    print("=" * 50)

    result = apply_kernel(img, 'sobel_x')
    print(f"\nApplied Sobel X to {img.name}")
    print(f"Result: {result}")

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
    print("\n" + "=" * 50)
    print("Example 4: Convolution Step-by-Step")
    print("=" * 50)

    img = create_sample_image(5, 'edges')
    print(f"\nImage ({img.height}x{img.width}):")
    matrix = img.as_matrix()
    for row in matrix:
        print("  " + " ".join(f"{v:.1f}" for v in row))

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
    print("\n" + "=" * 50)
    print("Example 5: Affine Transformations")
    print("=" * 50)

    transform = AffineTransform()
    transform.rotate(30)

    print("\nRotation matrix (30 degrees):")
    print(transform.matrix)

    original = (1, 0)
    new_x, new_y = transform.transform_point(*original)
    print(f"\nPoint {original} rotated 30 degrees:")
    print(f"  New position: ({new_x:.3f}, {new_y:.3f})")

    transform2 = AffineTransform()
    transform2.rotate(15).scale(2.0).translate(1, 1)
    print("\nComposed transformation (rotate 15 + scale 2x + translate):")
    print(transform2.matrix)


def example_ml_cv_pipeline():
    print("\n" + "=" * 50)
    print("Example 6: ML/CV Pipeline")
    print("=" * 50)

    img = create_sample_image(16, 'circle')
    print(f"\n1. Input: {img}")

    normalized = normalize_image(img)
    print(f"2. Normalized: mean={np.mean(normalized.data):.3f}, std={np.std(normalized.data):.3f}")

    edges = apply_kernel(normalized, 'sobel_x')
    print(f"3. Edge detection: {edges}")

    smoothed = apply_kernel(edges, 'gaussian_blur')
    print(f"4. Blur applied: {smoothed}")

    print(f"\nOperation history: {edges.history}")


def run_examples():
    img = example_image_as_matrix()
    example_convolution_kernels()
    example_apply_convolution(img)
    example_convolution_step_by_step()
    example_affine_transforms()
    example_ml_cv_pipeline()
