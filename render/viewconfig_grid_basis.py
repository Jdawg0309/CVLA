"""
Grid basis helpers for view configuration.
"""

import numpy as np


def grid_axes(self):
    """Return indices of axes that form the grid plane."""
    if self.grid_plane == "xy":
        return (self.axis_map['x'], self.axis_map['y'])
    elif self.grid_plane == "xz":
        return (self.axis_map['x'], self.axis_map['z'])
    elif self.grid_plane == "yz":
        return (self.axis_map['y'], self.axis_map['z'])


def get_grid_normal(self):
    """Get the normal vector of the grid plane."""
    if self.grid_plane == "xy":
        return np.array([0, 0, 1], dtype=np.float32)
    elif self.grid_plane == "xz":
        return np.array([0, 1, 0], dtype=np.float32)
    else:
        return np.array([1, 0, 0], dtype=np.float32)


def get_grid_basis(self):
    """Get basis vectors for the grid plane."""
    axes = self.grid_axes()

    basis = []
    for i in range(3):
        vec = np.zeros(3, dtype=np.float32)
        if i in axes:
            vec[i] = 1.0
        basis.append(vec)

    return basis
