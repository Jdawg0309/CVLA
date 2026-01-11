"""
Axis mapping helpers for view configuration.
"""

import numpy as np


def _setup_axis_mapping(self):
    """Setup axis mapping based on up axis."""
    if self.up_axis == "z":
        self.axis_map = {
            'x': 0, 'y': 1, 'z': 2,
            'right': 0, 'forward': 1, 'up': 2
        }
        self.axis_names = {'x': 'X', 'y': 'Y', 'z': 'Z'}

    elif self.up_axis == "y":
        self.axis_map = {
            'x': 0, 'y': 2, 'z': 1,
            'right': 0, 'forward': 2, 'up': 1
        }
        self.axis_names = {'x': 'X', 'y': 'Z', 'z': 'Y'}

    else:
        self.axis_map = {
            'x': 1, 'y': 2, 'z': 0,
            'right': 1, 'forward': 2, 'up': 0
        }
        self.axis_names = {'x': 'Y', 'y': 'Z', 'z': 'X'}


def axis_vectors(self):
    """Get world-space axis vectors based on up_axis."""
    if self.up_axis == "z":
        return {
            "x": np.array([1, 0, 0], dtype=np.float32),
            "y": np.array([0, 1, 0], dtype=np.float32),
            "z": np.array([0, 0, 1], dtype=np.float32)
        }
    elif self.up_axis == "y":
        return {
            "x": np.array([1, 0, 0], dtype=np.float32),
            "y": np.array([0, 0, 1], dtype=np.float32),
            "z": np.array([0, 1, 0], dtype=np.float32)
        }
    else:
        return {
            "x": np.array([0, 1, 0], dtype=np.float32),
            "y": np.array([0, 0, 1], dtype=np.float32),
            "z": np.array([1, 0, 0], dtype=np.float32)
        }


def axis_label_strings(self):
    """Get descriptive axis labels."""
    return {
        "x": self.axis_names['x'],
        "y": self.axis_names['y'],
        "z": self.axis_names['z']
    }
