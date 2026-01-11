"""
CVLA - Complete Visual Linear Algebra
Main entry point for the enhanced 3D linear algebra visualizer

This application demonstrates linear algebra concepts visually, including:
- Vector operations (add, scale, dot product, cross product)
- Matrix transformations
- Linear systems of equations
- Image processing as matrix operations (ML/CV visualization)
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runtime.app import App


def demo_image_processing():
    """
    Demonstrate how images are matrices and how convolutions work.

    This example shows the fundamental operations used in CNNs:
    - Loading/creating images as matrices
    - Applying convolution kernels
    - Affine transformations

    Run with: python main.py --demo-vision
    """
    try:
        from vision import (
            create_sample_image, apply_kernel, list_kernels,
            get_kernel_by_name, AffineTransform, apply_affine_transform,
            compute_gradient_magnitude
        )
        import numpy as np
    except ImportError as e:
        print(f"Vision module not available: {e}")
        print("Install Pillow: pip install Pillow")
        return

    print("=" * 60)
    print("CVLA Vision Demo: Images as Matrices")
    print("=" * 60)
    print()

    # Step 1: Create a sample image (checkerboard pattern)
    print("Step 1: Creating a checkerboard image (8x8)")
    print("-" * 40)
    img = create_sample_image(8, 'checkerboard')
    print(f"Image shape: {img.shape}")
    print(f"Image type: {'Grayscale' if img.is_grayscale else 'RGB'}")
    print()
    print("Matrix representation (pixel values):")
    matrix = img.as_matrix()
    for row in matrix:
        print("  " + " ".join(f"{v:.1f}" for v in row))
    print()

    # Step 2: Show available kernels
    print("Step 2: Convolution Kernels (feature detectors)")
    print("-" * 40)
    print("Available kernels:")
    for name, desc in list_kernels()[:6]:
        print(f"  - {name}: {desc}")
    print()

    # Step 3: Show a kernel's structure
    print("Step 3: Sobel X kernel (vertical edge detector)")
    print("-" * 40)
    sobel_x = get_kernel_by_name('sobel_x')
    print("Kernel matrix:")
    for row in sobel_x:
        print("  " + " ".join(f"{v:>5.1f}" for v in row))
    print()
    print("This kernel detects vertical edges by computing")
    print("the difference between left and right pixel values.")
    print()

    # Step 4: Apply convolution
    print("Step 4: Applying Sobel X to checkerboard")
    print("-" * 40)
    result = apply_kernel(img, 'sobel_x', normalize_output=True)
    print(f"Output shape: {result.shape}")
    print()
    print("Result matrix (edge response):")
    result_matrix = result.as_matrix()
    for row in result_matrix:
        print("  " + " ".join(f"{v:.2f}" for v in row))
    print()
    print("High values indicate detected edges!")
    print()

    # Step 5: Affine transformation
    print("Step 5: Affine Transformation (rotation)")
    print("-" * 40)
    transform = AffineTransform()
    transform.rotate(45, center=(4, 4))  # 45 degree rotation
    print("Rotation matrix (45 degrees):")
    for row in transform.matrix:
        print("  " + " ".join(f"{v:>6.3f}" for v in row))
    print()
    print("This matrix maps each pixel coordinate to its new position.")
    print()

    # Step 6: ML/CV insight
    print("=" * 60)
    print("ML/CV Insight")
    print("=" * 60)
    print("""
In Convolutional Neural Networks (CNNs):
1. Images are treated as matrices (exactly like above)
2. Kernels are LEARNED from data (not hand-designed)
3. Multiple kernels are applied in parallel = "feature maps"
4. The network learns what features are useful for the task

The Sobel kernel we used manually is similar to what
neural networks learn automatically in early layers.
Deep networks stack many such operations to recognize
complex patterns like faces, objects, and scenes.

Key insight: Every image operation is just matrix math!
    """)


def main():
    """Main entry point for CVLA application."""
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--demo-vision':
            demo_image_processing()
            return
        elif sys.argv[1] == '--help':
            print("CVLA - Complete Visual Linear Algebra")
            print()
            print("Usage:")
            print("  python main.py           Launch the interactive application")
            print("  python main.py --demo-vision  Run image processing demo")
            print("  python main.py --help    Show this help message")
            return

    try:
        app = App()
        app.run()
    except Exception as e:
        print(f"Error starting CVLA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
