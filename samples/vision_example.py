#!/usr/bin/env python3
"""
CVLA Vision Example: Image Processing as Linear Algebra
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from samples.vision_examples import run_examples


def main():
    print("=" * 60)
    print("  CVLA Vision Module: Image Processing as Linear Algebra")
    print("=" * 60)
    print("""
    This example demonstrates how images and convolutions work
    at the mathematical level - the foundation of CNNs and ML.
    """)

    run_examples()

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
